use std::{
    collections::HashSet,
    io::{BufRead, Seek},
    path::{Path, PathBuf},
    process::Command,
};

use rayon::slice::ParallelSliceMut;

use crate::*;

struct ParseOut {
    traces: Vec<MpiTrace>,
    missing_mpiops: HashSet<String>,
}

pub struct DumpiTxtParser<'init, 'update> {
    app_trace_dir: PathBuf,
    init_fn: Box<dyn Fn(u64) + Sync + 'init>,
    update_fn: Box<dyn Fn() + Sync + 'update>,
}

impl<'init, 'update> DumpiTxtParser<'init, 'update> {
    pub fn new(
        app_trace_dir: &Path,
        init_fn: impl Fn(u64) + Sync + 'init,
        update_fn: impl Fn() + Sync + 'update,
    ) -> Self {
        Self {
            app_trace_dir: app_trace_dir.to_owned(),
            init_fn: Box::new(init_fn),
            update_fn: Box::new(update_fn),
        }
    }
}

impl<'init, 'update> crate::TraceParser for DumpiTxtParser<'init, 'update> {
    fn parse_trace(&self) -> MpiAppTrace {
        let mut dir_entries = std::fs::read_dir(&self.app_trace_dir)
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        dir_entries.sort();

        let mut metadata = None;

        let mut bin_traces = Vec::new();
        for path in dir_entries.into_iter().rev() {
            if let Some(ext) = path.extension() {
                match ext.to_str().unwrap() {
                    "meta" => {
                        metadata = Some(parse_metadata(path));
                        continue;
                    }
                    "bin" => {
                        // metadata should be the first parsed file, so unwraping here is fine
                        let prefix = &metadata.as_ref().unwrap().prefix;
                        let filename = path.file_name().unwrap().to_str().unwrap();

                        // gets `my_rank` to the one DUMPI provides in the file name
                        let diff = filename.strip_prefix(prefix).unwrap();
                        let my_rank = sscanf::sscanf!(diff, "-{u32}.bin").unwrap();

                        bin_traces.push((path, my_rank));
                    }
                    _ => continue,
                }
            }
        }

        (self.init_fn)(bin_traces.len() as u64);

        // Debugger crashes with rayon par_iter
        #[cfg(debug_assertions)]
        let (mut dumps, missing_mpiops): (Vec<(u32, Vec<MpiTrace>)>, HashSet<String>) = bin_traces
            .iter()
            .map(|(job_path, job_rank)| {
                let parsed_file = parse_dump(job_path);
                (self.update_fn)();
                ((*job_rank, parsed_file.traces), parsed_file.missing_mpiops)
            })
            .fold(
                (Vec::new(), HashSet::new()),
                |(mut dumps, mut missing_mpiops), (dump, mpiops)| {
                    dumps.push(dump);
                    missing_mpiops.extend(mpiops);
                    (dumps, missing_mpiops)
                },
            );

        #[cfg(not(debug_assertions))]
        let (mut dumps, missing_mpiops) = {
            use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
            use std::sync::Mutex;

            let aggr_info = bin_traces
                .par_iter()
                .map(|job| {
                    let parsed_file = parse_dump(&job.0);
                    (self.update_fn)();
                    ((job.1, parsed_file.traces), parsed_file.missing_mpiops)
                })
                .fold(
                    || (Mutex::new(Vec::new()), Mutex::new(HashSet::new())),
                    |(lhs_dumps, lhs_missings), (rhs_dump, rhs_miss)| {
                        lhs_dumps.lock().unwrap().push(rhs_dump);
                        lhs_missings.lock().unwrap().extend(rhs_miss);

                        (lhs_dumps, lhs_missings)
                    },
                )
                .reduce(
                    || (Mutex::new(Vec::new()), Mutex::new(HashSet::new())),
                    |(lhs_dumps, lhs_missings), (rhs_dump, rhs_miss)| {
                        lhs_dumps
                            .lock()
                            .unwrap()
                            .extend(rhs_dump.into_inner().unwrap());

                        lhs_missings
                            .lock()
                            .unwrap()
                            .extend(rhs_miss.into_inner().unwrap());

                        (lhs_dumps, lhs_missings)
                    },
                );

            (
                aggr_info.0.into_inner().unwrap(),
                aggr_info.1.into_inner().unwrap(),
            )
        };

        dumps.par_sort_unstable_by_key(|(rank, _)| *rank);

        MpiAppTrace {
            metadata: metadata.unwrap(),
            dumps,
            missing_mpiops,
        }
    }
}

fn parse_metadata(path: impl AsRef<std::path::Path>) -> MpiAppMeta {
    let mut output = MpiAppMeta {
        num_procs: 0,
        start_time: 0.0,
        prefix: String::new(),
    };

    let text = std::fs::read_to_string(path).unwrap();
    for line in text.lines() {
        let mut split = line.split('=');
        let key = split.next().unwrap();
        let value = split.next_back().unwrap();

        match key {
            "numprocs" => output.num_procs = value.parse().unwrap(),
            "startime" => output.start_time = value.parse().unwrap(),
            "fileprefix" => output.prefix = value.to_string(),
            _ => {}
        }
    }

    output
}

fn parse_dump(path: &std::path::Path) -> ParseOut {
    let mut tmp_file = tempfile::tempfile().unwrap();

    Command::new("dumpi2ascii")
        .arg(path)
        .stdout(std::process::Stdio::from(tmp_file.try_clone().unwrap()))
        .output()
        .expect("\"dumpi2ascii\" could not be found. It is installed?");

    // Seek the fd back to the start after writing
    tmp_file.seek(std::io::SeekFrom::Start(0)).unwrap();

    let dumpi_reader = std::io::BufReader::new(tmp_file);
    let mut parsed_traces = Vec::new();

    #[derive(Clone, Copy)]
    enum ParsingState {
        Entering,
        Returning,
    }

    fn parse_dumpi_state(state_str: &str) -> ParsingState {
        match state_str {
            "entering" => ParsingState::Entering,
            "returning" => ParsingState::Returning,
            inv_state => panic!("Invalid DUMPI operation state: {inv_state}"),
        }
    }

    struct TraceData {
        start_wtime: f64,
        end_wtime: f64,
        thread_id: u32,
    }

    let mut current_state = None;
    let mut current_trace_data = None;
    let mut current_trace_state = None;
    let mut current_trace_op: Option<MpiOp> = None;
    let mut missing_mpiops = HashSet::new();

    for line in dumpi_reader.lines() {
        let line = line.unwrap();
        let line = line.trim();

        let parse = sscanf::sscanf!(
            line,
            "{str} {str} at walltime {f64}, cputime {f64} seconds in thread {u32}."
        );

        match parse {
            Ok((op_str, state_str, wtime, _cpu_time, thread_id)) => {
                current_state = Some(parse_dumpi_state(state_str));

                if let (Some(ParsingState::Returning), Some(op)) =
                    (current_state, &current_trace_op)
                {
                    // Takes from `current_args` and clears with `None`
                    // Shouldn't be `None` before this
                    let mut args: TraceData = current_trace_data.take().unwrap();
                    args.end_wtime = wtime;

                    parsed_traces.push(MpiTrace {
                        op: op.clone(),
                        start_wtime: args.start_wtime,
                        end_wtime: args.end_wtime,
                        thread_id: args.thread_id,
                    })
                }

                current_trace_op = match_mpi_op(op_str, &mut missing_mpiops);
                current_trace_state = Some(Default::default());

                if current_trace_op.is_some() && current_trace_data.is_none() {
                    current_trace_data = Some(TraceData {
                        start_wtime: wtime,
                        end_wtime: 0.0,
                        thread_id,
                    })
                }
            }
            Err(e) => match e {
                sscanf::Error::MatchFailed => {
                    if let (
                        Some(ParsingState::Entering),
                        Some(ref mut trace),
                        Some(ref mut preserve_state),
                    ) = (
                        current_state,
                        &mut current_trace_op,
                        &mut current_trace_state,
                    ) {
                        parse_mpi_args(line, trace, preserve_state);
                    }
                }
                sscanf::Error::ParsingFailed(pe) => panic!("DUMPI parsing failed! {pe}",),
            },
        }
    }

    ParseOut {
        traces: parsed_traces,
        missing_mpiops,
    }
}

fn match_mpi_op(op_str: &str, missing_mpiops: &mut HashSet<String>) -> Option<MpiOp> {
    match op_str {
        // Sends
        "MPI_Send" => Some(MpiOp::Send(Default::default())),
        "MPI_Isend" => Some(MpiOp::Isend(Default::default())),
        // Recvs
        "MPI_Recv" => Some(MpiOp::Recv(Default::default())),
        "MPI_Irecv" => Some(MpiOp::Irecv(Default::default())),
        // Tests
        "MPI_Test" => todo!("{op_str}"),
        "MPI_Testall" => todo!("{op_str}"),
        "MPI_Testany" => todo!("{op_str}"),
        "MPI_Testsome" => todo!("{op_str}"),
        // Waits
        "MPI_Wait" => Some(MpiOp::Wait { request_id: 0 }),
        "MPI_Waitall" => Some(MpiOp::Waitall {
            requests_ids: Vec::new(),
        }),
        "MPI_Waitany" => Some(MpiOp::Waitany { request_id: 0 }),
        "MPI_Waitsome" => Some(MpiOp::Waitsome {
            requests_ids: Vec::new(),
        }),
        // Collectives
        "MPI_Allgather" => Some(MpiOp::Allgather),
        "MPI_Allgatherv" => Some(MpiOp::Allgatherv),
        "MPI_Allreduce" => Some(MpiOp::Allreduce),
        "MPI_Alltoall" => Some(MpiOp::Alltoall),
        "MPI_Alltoallv" => Some(MpiOp::Alltoallv),
        "MPI_Alltoallw" => Some(MpiOp::Alltoallw),
        "MPI_Barrier" => Some(MpiOp::Barrier),
        "MPI_Bcast" => Some(MpiOp::Bcast),
        "MPI_Gather" => Some(MpiOp::Gather),
        "MPI_Gatherv" => Some(MpiOp::Gatherv),
        "MPI_Iallgather" => Some(MpiOp::Iallgather),
        "MPI_Iallreduce" => Some(MpiOp::Iallreduce),
        "MPI_Ibarrier" => Some(MpiOp::Ibarrier),
        "MPI_Ibcast" => Some(MpiOp::Ibcast),
        "MPI_Iagther" => Some(MpiOp::Iagther),
        "MPI_Iagtherv" => Some(MpiOp::Iagtherv),
        "MPI_Ireduce" => Some(MpiOp::Ireduce),
        "MPI_Iscatter" => Some(MpiOp::Iscatter),
        "MPI_Iscatterv" => Some(MpiOp::Iscatterv),
        "MPI_Reduce" => Some(MpiOp::Reduce),
        "MPI_Scatter" => Some(MpiOp::Scatter),
        "MPI_Scatterv" => Some(MpiOp::Scatterv),
        _ => {
            missing_mpiops.insert(op_str.to_string());
            None
        }
    }
}

#[derive(Default)]
struct MpiPerserveState {
    // For Waitany
    index: u32,
    // For Waitsome
    indices: Vec<u32>,
}

fn parse_mpi_args(line: &str, op: &mut MpiOp, state: &mut MpiPerserveState) {
    let (_info_token_lhs, info_token_rhs) = line.split_once(' ').unwrap();
    let mut tokens = info_token_rhs.split('=');

    let key = tokens.next().unwrap();
    let value = tokens.next().unwrap();

    // <IGNORED> is MPI_STATUS_IGNORE or MPI_STATUSES_IGNORE and any other kind of ignore I guess
    if value == "<IGNORED>" {
        return;
    }

    enum ParsingArgs {
        Count(u32),
        Dtty(u32),
        Src(Option<u32>),
        Dest(Option<u32>),
        Tag(Option<u32>),
        Comm(MpiComm),
        Request(u32),
        Requests(Vec<u32>),
        Index(u32),
        Indices(Vec<u32>),
    }

    let args = match key {
        "count" => Some(ParsingArgs::Count(value.parse().unwrap())),
        "datatype" => {
            let mut dtty_tokens = value.split_whitespace();
            Some(ParsingArgs::Dtty(
                dtty_tokens.next().unwrap().parse().unwrap(),
            ))
        }
        // Not using value.parse.ok() because None means MPI_ANY_...
        "source" => Some(ParsingArgs::Src(value.parse().ok())),
        "dest" => Some(ParsingArgs::Dest(Some(value.parse().unwrap()))),
        "tag" => Some(ParsingArgs::Tag(value.parse().ok())),
        "comm" => {
            let mut comm_tokens = value.split_whitespace();
            Some(ParsingArgs::Comm(
                match comm_tokens.next().unwrap().parse::<u32>().unwrap() {
                    2 => MpiComm::World,
                    other => MpiComm::Other(other),
                },
            ))
        }
        "request" => Some(ParsingArgs::Request(
            value
                .trim_matches(|c| c == '[' || c == ']')
                .parse()
                .unwrap_or_else(|_| panic!("Cannot parse `{value}` as MPI_Request")),
        )),
        _ if key.starts_with("requests") => Some(ParsingArgs::Requests(
            value
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|n| n.trim().parse())
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| panic!("Cannot parse `{value}` as MPI_Request list")),
        )),
        "index" => Some(ParsingArgs::Index(value.parse().unwrap())),
        _ if key.starts_with("indices") => Some(ParsingArgs::Indices(
            value
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .map(|n| n.trim().parse())
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|_| panic!("Cannot parse `{value}` as indices")),
        )),
        _ => None,
    };

    if let Some(args) = args {
        let is_async = matches!(op, MpiOp::Isend(_) | MpiOp::Irecv(_));

        match op {
            MpiOp::Send(sinfo) | MpiOp::Isend(sinfo) => match args {
                ParsingArgs::Count(count) => sinfo.count = count,
                ParsingArgs::Dtty(dtty) => sinfo.dtty = dtty,
                ParsingArgs::Dest(dest) => sinfo.dest_rank = dest,
                ParsingArgs::Tag(tag) => sinfo.tag = tag,
                ParsingArgs::Comm(comm) => sinfo.comm = comm,
                ParsingArgs::Request(request) => {
                    sinfo.request = if is_async { Some(request) } else { None }
                }
                _ => (),
            },
            MpiOp::Recv(rinfo) | MpiOp::Irecv(rinfo) => match args {
                ParsingArgs::Count(count) => rinfo.count = count,
                ParsingArgs::Dtty(dtty) => rinfo.dtty = dtty,
                ParsingArgs::Src(src) => rinfo.src_rank = src,
                ParsingArgs::Tag(tag) => rinfo.tag = tag,
                ParsingArgs::Comm(comm) => rinfo.comm = comm,
                ParsingArgs::Request(request) => {
                    rinfo.request = if is_async { Some(request) } else { None }
                }
                _ => (),
            },
            //Tests
            MpiOp::Test {} => todo!(),
            MpiOp::Testall {} => todo!(),
            MpiOp::Testany {} => todo!(),
            MpiOp::Testsome {} => todo!(),
            // Waits
            MpiOp::Wait { request_id } => {
                if let ParsingArgs::Request(request) = args {
                    *request_id = request;
                }
            }
            MpiOp::Waitall { requests_ids } => {
                if let ParsingArgs::Requests(requests) = args {
                    *requests_ids = requests;
                }
            }
            MpiOp::Waitany {
                request_id: requests_id,
            } => {
                if let ParsingArgs::Index(index) = args {
                    state.index = index;
                } else if let ParsingArgs::Requests(requests) = args {
                    *requests_id = requests[state.index as usize];
                }
            }
            MpiOp::Waitsome { requests_ids } => {
                // Indicies MUST be parsed before requests
                // Using `MpiPreserveState` struct for that
                if let ParsingArgs::Indices(indices) = args {
                    state.indices = indices;
                } else if let ParsingArgs::Requests(mut requests) = args {
                    for rm_id in &state.indices {
                        requests.remove(*rm_id as usize);
                    }
                    *requests_ids = requests;
                }
            }
            _ => {}
        }
    }
}
