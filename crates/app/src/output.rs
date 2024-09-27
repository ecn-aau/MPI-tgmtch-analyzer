use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use traceback::RankSimOut;
use traceparsers::{MpiAppTrace, MpiOp, MpiOpKind};

const STATS_FILE_NAME: &str = "stats.csv";
const TAGS_FILE_NAME: &str = "tags.csv";
const RECVS_FILE_NAME: &str = "recvs.csv";

pub struct Output<'traces> {
    root_path: PathBuf,
    traces: &'traces BTreeMap<u32, RankSimOut>,
}

impl<'traces> Output<'traces> {
    pub fn new(
        root_path: &std::path::Path,
        traces: &'traces BTreeMap<u32, RankSimOut>,
    ) -> Result<Self, std::io::Error> {
        std::fs::create_dir(root_path)?;

        Ok(Self {
            root_path: root_path.to_owned(),
            traces,
        })
    }

    pub fn write_output(&self) {
        std::thread::scope(|s| {
            s.spawn(|| self.write_stats());
            s.spawn(|| self.write_tag_dist());
        });
    }

    fn write_stats(&self) {
        let mut stats_writer = csv::WriterBuilder::new()
            .from_path(self.root_path.join(STATS_FILE_NAME))
            .unwrap();

        self.traces.iter().for_each(|(_, val)| {
            val.timeline.iter().for_each(|(_, stat)| {
                stats_writer.serialize(stat).unwrap();
            });
        });
    }

    fn write_tag_dist(&self) {
        let mut tag_writer = csv::WriterBuilder::new()
            .from_path(self.root_path.join(TAGS_FILE_NAME))
            .unwrap();

        // Header values (beside rank, op type and mode)
        let mut used_tags = BTreeSet::<Option<u32>>::new();

        // Map<Rank, Timeline<Map<Tag, Number of times used>>>
        // Order: Send sync, send async, recv sync, recv async
        let mut tags_values =
            std::iter::repeat_with(BTreeMap::<u32, Vec<BTreeMap<Option<u32>, u32>>>::new)
                .take(4)
                .collect::<Vec<_>>();

        self.traces.iter().for_each(|(rank, sim_out)| {
            for (_, stat) in sim_out.timeline.iter() {
                let tags = [
                    &stat.sends_per_tag_sync,
                    &stat.sends_per_tag_async,
                    &stat.recvs_per_tag_sync,
                    &stat.recvs_per_tag_async,
                ];

                used_tags.extend(tags.iter().flat_map(|tag_info| tag_info.keys()));
                tags_values
                    .iter_mut()
                    .zip(tags)
                    .for_each(|(tag_value, tag_data)| {
                        tag_value
                            .entry(*rank)
                            .and_modify(|value| value.push(tag_data.clone()))
                            .or_default();
                    });
            }
        });

        // Write headers [rank, op (send, recv), kind (sync, async), tags...]
        tag_writer
            .write_record(["rank", "op", "kind"].map(String::from).into_iter().chain(
                used_tags.iter().map(|entry| match entry {
                    Some(tag) => tag.to_string(),
                    None => String::from("wildcard"),
                }),
            ))
            .unwrap();

        let ops = [String::from("send"), String::from("recv")];
        let kinds = [String::from("sync"), String::from("async")];

        // Write values
        for (id, map) in tags_values.iter().enumerate() {
            let (op, kind) = (&ops[id / 2], &kinds[id % 2]);
            for (rank, values) in map {
                values
                    .iter()
                    .map(|val| {
                        used_tags.iter().map(|key| match val.get(key) {
                            Some(n) => n.to_string(),
                            None => String::from("0"),
                        })
                    })
                    .for_each(|row| {
                        tag_writer
                            .write_record(
                                [rank.to_string(), op.clone(), kind.clone()]
                                    .into_iter()
                                    .chain(row),
                            )
                            .unwrap();
                    });
            }
        }
    }

    pub fn write_recvs_info(&self, trace: &MpiAppTrace) {
        let mut recv_writer = csv::WriterBuilder::new()
            .from_path(self.root_path.join(RECVS_FILE_NAME))
            .unwrap();

        // Write headers [rank, src, tag]
        recv_writer.write_record(["rank", "src", "tag"]).unwrap();

        for (rank, timeline) in trace.dumps() {
            for call in timeline {
                if let MpiOp::Recv(recv) | MpiOp::Irecv(recv) = call.op {
                    let src = match recv.src_rank {
                        Some(src) => src.to_string(),
                        None => "none".to_string(),
                    };

                    let tag = match recv.tag {
                        Some(tag) => tag.to_string(),
                        None => "none".to_string(),
                    };

                    recv_writer
                        .write_record([rank.to_string(), src, tag])
                        .unwrap();
                }
            }
        }
    }
}
