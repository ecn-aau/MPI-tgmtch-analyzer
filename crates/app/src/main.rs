use std::{fs::File, io::BufReader, num::NonZeroUsize, ops::RangeInclusive, str::FromStr};

use clap::Parser;
use traceparsers::{MpiAppTrace, TraceParser};

mod output;

const BINS_RANGE: RangeInclusive<usize> = 1..=256;
const CACHE_FILENAME: &str = "mts.cache";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// Path to a directory with a DUMPI trace for a single application.
    #[arg(short = 't', long = "trace", value_name = "DIR")]
    trace_dir: std::path::PathBuf,
    /// Number of statistic points reduced into a single one.
    #[arg(short = 'g', long = "groups", value_name = "GROUPS")]
    group_size: Option<NonZeroUsize>,
    /// Number of BINS (1...256) for the no wildcards hash table.
    #[arg(long = "nb", value_name = "BINS", value_parser = bin_in_range)]
    no_wild_bins: usize,
    /// Number of BINS (1...256) for the MPI_ANY_SRC hash table.
    #[arg(long = "sb", value_name = "BINS", value_parser = bin_in_range)]
    src_wild_bins: usize,
    /// Number of BINS (1...256) for the MPI_ANY_TAG hash table.
    #[arg(long = "tb", value_name = "BINS", value_parser = bin_in_range)]
    tag_wild_bins: usize,
    /// Name for the output folder. If the folder exists, program ends.
    ///
    /// If not specifed, no output is written. Mean for development.
    #[arg(short = 'o', long = "output", value_name = "FOLDER", value_parser = valid_path)]
    output: Option<std::path::PathBuf>,
    /// Regenerates traces cache file (`mts.cache` on trace folder).
    ///
    /// Useful new using a new version of the software.
    #[arg(short = 'f', long = "force")]
    force_regen_cache: bool,
    /// Display not implemented MPI operations in the parser and/or simulator.
    ///
    /// Useful for development
    #[arg(long = "missing")]
    show_missing: bool,
}

fn bin_in_range(s: &str) -> Result<usize, String> {
    let size: usize = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a valid BIN size"))?;
    if BINS_RANGE.contains(&size) {
        Ok(size)
    } else {
        Err(format!(
            "BIN is not in range {}-{}",
            BINS_RANGE.start(),
            BINS_RANGE.end()
        ))
    }
}

fn valid_path(s: &str) -> Result<std::path::PathBuf, String> {
    let path = std::path::PathBuf::from_str(s).map_err(|_| "FOLDER is not valid".to_string())?;
    if !path.exists() || !path.is_file() {
        Ok(path)
    } else {
        Err("FOLDER is not a valid (duplicate? invalid?)".to_string())
    }
}

fn get_cache_for_trace(cache_path: &std::path::Path) -> Option<MpiAppTrace> {
    let cache_file = File::open(cache_path).ok()?;
    let buf_reader = BufReader::new(cache_file);
    let mut decoder = snap::read::FrameDecoder::new(buf_reader);

    bincode::decode_from_std_read(&mut decoder, bincode::config::standard()).ok()
}

fn main() {
    let args = CliArgs::parse();

    let parsing_pb = indicatif::ProgressBar::new(0)
        .with_message("Parsing traces")
        .with_style(indicatif::ProgressStyle::with_template("{spinner:.green} [{elapsed:>3}] [{msg}] [{wide_bar:.cyan/blue}] {percent:>2}% ({eta})")
        .unwrap()
        .with_key("eta", |state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));

    let parser = traceparsers::dumpi_txt::DumpiTxtParser::new(
        &args.trace_dir,
        |items| parsing_pb.inc_length(items),
        || parsing_pb.inc(1),
    );

    let cache_path = args.trace_dir.join(CACHE_FILENAME);
    let app_trace = if let (Some(trace), false) =
        (get_cache_for_trace(&cache_path), args.force_regen_cache)
    {
        trace
    } else {
        let trace: MpiAppTrace = parser.parse_trace();

        let cache_file = File::create(cache_path).unwrap();
        let mut encoder = snap::write::FrameEncoder::new(cache_file);

        bincode::encode_into_std_write(&trace, &mut encoder, bincode::config::standard()).unwrap();

        trace
    };
    parsing_pb.finish();

    let simulating_pb = indicatif::ProgressBar::new(0)
        .with_message("Simulating app")
        .with_style(indicatif::ProgressStyle::with_template("{spinner:.green} [{elapsed:>3}] [{msg}] [{wide_bar:.cyan/blue}] {percent:>2}% ({eta})")
        .unwrap()
        .with_key("eta", |state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));

    let newsim = traceback::TracebackSim::new(
        &app_trace,
        args.group_size
            .unwrap_or(NonZeroUsize::new(1).unwrap())
            .get(),
        args.no_wild_bins,
        args.src_wild_bins,
        args.tag_wild_bins,
        |items| simulating_pb.inc_length(items),
        || simulating_pb.inc(1),
    )
    .simulate::<og_hash::OgHashU8>();
    simulating_pb.finish();

    let spinner = indicatif::ProgressBar::new_spinner().with_message("Writing output data");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    if let Some(out) = args.output {
        match output::Output::new(&out, &newsim) {
            Ok(output) => std::thread::scope(|scope| {
                scope.spawn(|| {
                    output.write_output();
                });
                scope.spawn(|| {
                    output.write_recvs_info(&app_trace);
                });
            }),
            Err(e) => println!("[WARN] Could not create output. Reason: {e}"),
        }
    }

    spinner.finish();

    if args.show_missing {
        println!("\n\tMissing `MpiOps`: ");
        for op in app_trace.missing_mpiops() {
            println!("\t\t{op}");
        }
    }
}
