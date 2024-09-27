use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use traceparsers::{MpiAppTrace, MpiComm, MpiOp, MpiOpKind};

const MAX_OPS_WITHOUT_TIMESTAMP: u32 = 4000;

#[derive(Debug, Clone, Copy)]
enum Wilderness {
    NoWild { hash: usize },
    Src { hash: usize },
    Tag { hash: usize },
    Double,
}

struct TagMatchingModel {
    // Hashtables
    no_wildcard: Box<[BTreeSet<u32>]>,
    src_wildcard: Box<[BTreeSet<u32>]>,
    tag_wildcard: Box<[BTreeSet<u32>]>,
    // "Linked list"
    double_wildcard: BTreeSet<u32>,
    // Pending async requests
    pending_reqs: BTreeMap<u32, Wilderness>,
}

impl TagMatchingModel {
    fn track_pending_reqs(&mut self, completed_reqs: &[u32]) {
        self.pending_reqs
            .iter()
            .filter(|(k, _)| completed_reqs.contains(k))
            .for_each(|(k, v)| match v {
                Wilderness::NoWild { hash } => {
                    self.no_wildcard[hash % self.no_wildcard.len()].remove(k);
                }
                Wilderness::Src { hash } => {
                    self.src_wildcard[hash % self.src_wildcard.len()].remove(k);
                }
                Wilderness::Tag { hash } => {
                    self.tag_wildcard[hash % self.tag_wildcard.len()].remove(k);
                }
                Wilderness::Double => {
                    self.double_wildcard.remove(k);
                }
            });

        self.pending_reqs.retain(|k, _| !completed_reqs.contains(k));
    }

    fn is_empty_pending_reqs(&self) -> bool {
        self.pending_reqs.is_empty()
    }

    fn empty_buckets_perc(&self, stats: &mut TimestampStats) {
        let empty_bins_perc_no_wildcard = self
            .no_wildcard
            .iter()
            .filter(|collision_queue| collision_queue.is_empty())
            .count() as f32
            / self.no_wildcard.len() as f32;

        let empty_bins_perc_src_wildcard = self
            .src_wildcard
            .iter()
            .filter(|collision_queue| collision_queue.is_empty())
            .count() as f32
            / self.src_wildcard.len() as f32;

        let empty_bins_perc_tag_wildcard = self
            .tag_wildcard
            .iter()
            .filter(|collision_queue| collision_queue.is_empty())
            .count() as f32
            / self.tag_wildcard.len() as f32;

        stats.empty_bins_perc_no_wildcard = stats
            .empty_bins_perc_no_wildcard
            .min(empty_bins_perc_no_wildcard);
        stats.empty_bins_perc_src_wildcard = stats
            .empty_bins_perc_src_wildcard
            .min(empty_bins_perc_src_wildcard);
        stats.empty_bins_perc_tag_wildcard = stats
            .empty_bins_perc_tag_wildcard
            .min(empty_bins_perc_tag_wildcard);

        // stats.empty_bins_perc_no_wildcard = empty_bins_perc_no_wildcard;
        // stats.empty_bins_perc_src_wildcard = empty_bins_perc_src_wildcard;
        // stats.empty_bins_perc_tag_wildcard = empty_bins_perc_tag_wildcard;
    }
}

pub struct RankSimOut {
    pub rank: u32,
    pub timeline: Vec<(f64, TimestampStats)>,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct TimestampStats {
    rank: u32,
    // Collisions
    pub collisions_no_wildcard: usize,
    pub collisions_src_wildcard: usize,
    pub collisions_tag_wildcard: usize,
    // Aggregated collisions
    pub aggr_collisions_no_wildcard: usize,
    pub aggr_collisions_src_wildcard: usize,
    pub aggr_collisions_tag_wildcard: usize,
    // Statistics sends
    pub send_sync: usize,
    pub send_async: usize,
    #[serde(skip_serializing)]
    pub sends_per_tag_sync: BTreeMap<Option<u32>, u32>,
    #[serde(skip_serializing)]
    pub sends_per_tag_async: BTreeMap<Option<u32>, u32>,
    // Statistics recv
    pub recv_no_wildcard_sync: usize,
    pub recv_src_wildcard_sync: usize,
    pub recv_tag_wildcard_sync: usize,
    pub recv_double_wildcard_sync: usize,
    #[serde(skip_serializing)]
    pub recvs_per_tag_sync: BTreeMap<Option<u32>, u32>,
    pub recv_no_wildcard_async: usize,
    pub recv_src_wildcard_async: usize,
    pub recv_tag_wildcard_async: usize,
    pub recv_double_wildcard_async: usize,
    #[serde(skip_serializing)]
    pub recvs_per_tag_async: BTreeMap<Option<u32>, u32>,
    // Statistics lengths
    pub collisions_no_wildcard_max_len: usize,
    pub collisions_src_wildcard_max_len: usize,
    pub collisions_tag_wildcard_max_len: usize,
    pub collisions_double_wildcard_max_len: usize,
    pub empty_bins_perc_no_wildcard: f32,
    pub empty_bins_perc_src_wildcard: f32,
    pub empty_bins_perc_tag_wildcard: f32,
    // Other stuff
    pub number_rdma: usize,        // Count number of RDMA operations
    pub number_p2p: usize,         // Count number of p2p operations
    pub number_collectives: usize, // Count number of collectives operations
}

pub struct TracebackSim<'trace, 'init, 'update> {
    app_trace: &'trace MpiAppTrace,
    group_size: usize,
    no_wild_bins: usize,
    src_wild_bins: usize,
    tag_wild_bins: usize,
    // Info hooks
    init_fn: Box<dyn Fn(u64) + 'init>,
    update_fn: Box<dyn Fn() + 'update>,
}

impl<'trace, 'init, 'update> TracebackSim<'trace, 'init, 'update> {
    pub fn new(
        app_trace: &'trace MpiAppTrace,
        group_size: usize,
        no_wild_bins: usize,
        src_wild_bins: usize,
        tag_wild_bins: usize,
        init_fn: impl Fn(u64) + 'init,
        update_fn: impl Fn() + 'update,
    ) -> Self {
        Self {
            app_trace,
            group_size,
            no_wild_bins,
            src_wild_bins,
            tag_wild_bins,
            init_fn: Box::new(init_fn),
            update_fn: Box::new(update_fn),
        }
    }

    pub fn simulate<H: hash::ByteHash>(self) -> BTreeMap<u32, RankSimOut> {
        let mut rank_timestamps = BTreeMap::new();

        (self.init_fn)(self.app_trace.len() as u64);

        for (rank, rank_trace) in self.app_trace {
            let mut timeline = Vec::new();
            let mut current_timestamp = TimestampStats {
                rank: *rank,
                empty_bins_perc_no_wildcard: 1.0,
                empty_bins_perc_src_wildcard: 1.0,
                empty_bins_perc_tag_wildcard: 1.0,
                ..Default::default()
            };

            let mut mpi_model = TagMatchingModel {
                no_wildcard: vec![BTreeSet::new(); self.no_wild_bins].into_boxed_slice(),
                src_wildcard: vec![BTreeSet::new(); self.src_wild_bins].into_boxed_slice(),
                tag_wildcard: vec![BTreeSet::new(); self.tag_wild_bins].into_boxed_slice(),
                double_wildcard: BTreeSet::new(),
                pending_reqs: BTreeMap::new(),
            };

            (self.init_fn)(rank_trace.len() as u64);

            let mut first_time = None;
            let mut current_reduce_size = 0;
            let mut ops_no_new_timestamp = 0; // Count the number of instructions between Waits/Test (tests not implemented)
            for trace in rank_trace {
                let start_time = first_time.get_or_insert(trace.start_wtime);

                // Max length within the bins before we consume completed requests
                current_timestamp.collisions_no_wildcard_max_len =
                    current_timestamp.collisions_no_wildcard_max_len.max(
                        mpi_model
                            .no_wildcard
                            .iter()
                            .map(|bset| bset.len().saturating_sub(1))
                            .max()
                            .unwrap_or(0),
                    );
                current_timestamp.collisions_src_wildcard_max_len =
                    current_timestamp.collisions_src_wildcard_max_len.max(
                        mpi_model
                            .src_wildcard
                            .iter()
                            .map(|bset| bset.len().saturating_sub(1))
                            .max()
                            .unwrap_or(0),
                    );
                current_timestamp.collisions_tag_wildcard_max_len =
                    current_timestamp.collisions_tag_wildcard_max_len.max(
                        mpi_model
                            .tag_wildcard
                            .iter()
                            .map(|bset| bset.len().saturating_sub(1))
                            .max()
                            .unwrap_or(0),
                    );
                current_timestamp.collisions_double_wildcard_max_len = current_timestamp
                    .collisions_double_wildcard_max_len
                    .max(mpi_model.double_wildcard.len());

                // Signals when a new test point is reached -> add new timestamp into timeline
                // Also, popout request for the collision simulator
                let new_timestamp = match &trace.op {
                    // TODO: Tests
                    // Waits
                    MpiOp::Wait { request_id } | MpiOp::Waitany { request_id } => {
                        mpi_model.track_pending_reqs(&[*request_id]);
                        true
                    }
                    MpiOp::Waitall { requests_ids } | MpiOp::Waitsome { requests_ids } => {
                        mpi_model.track_pending_reqs(requests_ids);
                        true
                    }
                    _ => {
                        ops_no_new_timestamp += 1;
                        ops_no_new_timestamp == MAX_OPS_WITHOUT_TIMESTAMP // Arbitrary number
                    }
                };

                if new_timestamp {
                    current_reduce_size += 1;
                }

                if new_timestamp && current_reduce_size == self.group_size {
                    // Push statistics into timeline
                    timeline.push((trace.end_wtime - *start_time, current_timestamp));

                    // Clean up
                    ops_no_new_timestamp = 0;
                    current_reduce_size = 0;
                    current_timestamp = TimestampStats {
                        rank: *rank,
                        empty_bins_perc_no_wildcard: 1.0,
                        empty_bins_perc_src_wildcard: 1.0,
                        empty_bins_perc_tag_wildcard: 1.0,
                        ..Default::default()
                    };
                }

                fn process_op(
                    hash: usize,
                    wilderness: Wilderness,
                    table: &mut Box<[BTreeSet<u32>]>,
                    pending_reqs: &mut BTreeMap<u32, Wilderness>,
                    timestamp_collision: &mut usize,
                    aggr_timestamp_collisions: &mut usize,
                    mpi_op: &MpiOp,
                ) {
                    let idx = hash % table.len();
                    let is_collision = !table[idx].is_empty();

                    if is_collision {
                        *timestamp_collision += 1;
                        *aggr_timestamp_collisions += table[idx].len();
                    }

                    if let MpiOp::Irecv(rinfo) = mpi_op {
                        table[idx].insert(rinfo.request.unwrap());
                        pending_reqs.insert(rinfo.request.unwrap(), wilderness);
                    }
                }

                // Collision simulator
                if let Some(wilderness) = get_wilderness::<H>(&trace.op) {
                    // wilderness assures that MpiOp is either Recv or Irecv
                    match wilderness {
                        // To no wildcards hashtable
                        Wilderness::NoWild { hash } => {
                            process_op(
                                hash,
                                wilderness,
                                &mut mpi_model.no_wildcard,
                                &mut mpi_model.pending_reqs,
                                &mut current_timestamp.collisions_no_wildcard,
                                &mut current_timestamp.aggr_collisions_no_wildcard,
                                &trace.op,
                            );
                        }
                        // To MPI_ANY_SRC hashtable
                        Wilderness::Src { hash } => {
                            process_op(
                                hash,
                                wilderness,
                                &mut mpi_model.src_wildcard,
                                &mut mpi_model.pending_reqs,
                                &mut current_timestamp.collisions_src_wildcard,
                                &mut current_timestamp.aggr_collisions_src_wildcard,
                                &trace.op,
                            );
                        }
                        // To MPI_ANY_TAG hashtable
                        Wilderness::Tag { hash } => {
                            process_op(
                                hash,
                                wilderness,
                                &mut mpi_model.tag_wildcard,
                                &mut mpi_model.pending_reqs,
                                &mut current_timestamp.collisions_tag_wildcard,
                                &mut current_timestamp.aggr_collisions_tag_wildcard,
                                &trace.op,
                            );
                        }
                        // To double-wildcard linked list
                        Wilderness::Double => {
                            if let MpiOp::Irecv(rinfo) = &trace.op {
                                assert!(!mpi_model.double_wildcard.insert(rinfo.request.unwrap()));
                            }
                        }
                    }

                    // Calculate percentage of empty buckets
                    mpi_model.empty_buckets_perc(&mut current_timestamp);
                }

                // Normal statistics
                match &trace.op {
                    MpiOp::Send(args) => {
                        current_timestamp
                            .sends_per_tag_sync
                            .entry(args.tag)
                            .and_modify(|e| *e += 1)
                            .or_insert(1);

                        current_timestamp.send_sync += 1;
                    }
                    MpiOp::Isend(args) => {
                        current_timestamp
                            .sends_per_tag_async
                            .entry(args.tag)
                            .and_modify(|e| *e += 1)
                            .or_insert(1);

                        current_timestamp.send_async += 1;
                    }
                    MpiOp::Recv(args) => {
                        current_timestamp
                            .recvs_per_tag_sync
                            .entry(args.tag)
                            .and_modify(|e| *e += 1)
                            .or_insert(1);
                    }
                    MpiOp::Irecv(args) => {
                        current_timestamp
                            .recvs_per_tag_async
                            .entry(args.tag)
                            .and_modify(|e| *e += 1)
                            .or_insert(1);
                    }
                    _ => {}
                }

                if let Some(wilderness) = get_wilderness::<H>(&trace.op) {
                    match wilderness {
                        Wilderness::NoWild { hash: _ } => match trace.op {
                            MpiOp::Recv(_) => current_timestamp.recv_no_wildcard_sync += 1,
                            MpiOp::Irecv(_) => current_timestamp.recv_no_wildcard_async += 1,
                            _ => {}
                        },
                        Wilderness::Src { hash: _ } => match trace.op {
                            MpiOp::Recv(_) => current_timestamp.recv_src_wildcard_sync += 1,
                            MpiOp::Irecv(_) => current_timestamp.recv_src_wildcard_async += 1,
                            _ => {}
                        },
                        Wilderness::Tag { hash: _ } => match trace.op {
                            MpiOp::Recv(_) => current_timestamp.recv_tag_wildcard_sync += 1,
                            MpiOp::Irecv(_) => current_timestamp.recv_tag_wildcard_async += 1,
                            _ => {}
                        },
                        Wilderness::Double => match trace.op {
                            MpiOp::Recv(_) => current_timestamp.recv_double_wildcard_sync += 1,
                            MpiOp::Irecv(_) => current_timestamp.recv_double_wildcard_async += 1,
                            _ => {}
                        },
                    }
                }

                if let Some(op_kind) = trace.op.get_kind() {
                    match op_kind {
                        MpiOpKind::OneSided => current_timestamp.number_rdma += 1,
                        MpiOpKind::PointToPoint => current_timestamp.number_p2p += 1,
                        MpiOpKind::Collective => current_timestamp.number_collectives += 1,
                    }
                }

                (self.update_fn)();
            }

            // Last group missing by grouping size
            if current_reduce_size != 0 || ops_no_new_timestamp != 0 {
                timeline.push((first_time.unwrap(), current_timestamp));
            }

            // No need for a `bool` inside, since it uses Some and None anyway
            static WARNED_ONCE: OnceLock<()> = OnceLock::new();
            if WARNED_ONCE.get().is_none() && !mpi_model.is_empty_pending_reqs() {
                WARNED_ONCE.get_or_init(|| {
                    println!("[WARN] Async is not done! Some requests are unmatched!");
                });
            }

            rank_timestamps.insert(
                *rank,
                RankSimOut {
                    rank: *rank,
                    timeline,
                },
            );
        }

        rank_timestamps
    }
}

fn get_wilderness<H: hash::ByteHash>(op: &MpiOp) -> Option<Wilderness> {
    fn inner_wilderness(src_wild: bool, tag_wild: bool, hash: usize) -> Wilderness {
        match (src_wild, tag_wild) {
            (true, true) => Wilderness::Double,
            (true, false) => Wilderness::Src { hash },
            (false, true) => Wilderness::Tag { hash },
            (false, false) => Wilderness::NoWild { hash },
        }
    }

    match op {
        MpiOp::Recv(rinfo) | MpiOp::Irecv(rinfo) => {
            // [src, tag, comm]
            let key_buf = [
                rinfo.src_rank.map_or(0, |v| v),
                rinfo.tag.map_or(0, |v| v),
                match rinfo.comm {
                    MpiComm::World => 2,
                    MpiComm::Other(o) => o,
                },
            ];
            let key_buf: [u8; 12] = unsafe { std::mem::transmute(key_buf) };
            let hash = H::hash_u8(&key_buf);

            Some(inner_wilderness(
                rinfo.src_rank.is_none(),
                rinfo.tag.is_none(),
                hash as usize,
            ))
        }
        _ => None,
    }
}
