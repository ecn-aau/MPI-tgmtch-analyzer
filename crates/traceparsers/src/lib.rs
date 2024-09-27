use std::collections::HashSet;

use bincode::{Decode, Encode};

pub mod dumpi_txt;

#[derive(Encode, Decode)]
pub struct MpiAppTrace {
    metadata: MpiAppMeta,
    dumps: Vec<(u32, Vec<MpiTrace>)>,
    /// Missing `MpiOps`. Helpful for development.
    ///
    /// Empty does not mean that all MPI instructions are implemented in the simulator.
    missing_mpiops: HashSet<String>,
}

impl MpiAppTrace {
    pub fn missing_mpiops(&self) -> impl Iterator<Item = &String> {
        self.missing_mpiops.iter()
    }

    pub fn len(&self) -> usize {
        self.dumps().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'dumps> IntoIterator for &'dumps MpiAppTrace {
    type Item = &'dumps (u32, Vec<MpiTrace>);
    type IntoIter = std::slice::Iter<'dumps, (u32, Vec<MpiTrace>)>;

    fn into_iter(self) -> Self::IntoIter {
        self.dumps.iter()
    }
}

#[derive(Encode, Decode)]
pub struct MpiAppMeta {
    pub num_procs: u32,
    pub start_time: f64,
    pub prefix: String,
}

#[derive(Encode, Decode)]
pub struct MpiTrace {
    pub op: MpiOp,
    pub start_wtime: f64,
    pub end_wtime: f64,
    pub thread_id: u32,
}

#[derive(Debug, Clone, Encode, Decode)]
pub enum MpiOp {
    // Sends
    Send(MpiSendInfo),
    Isend(MpiSendInfo),
    // Recvs
    Recv(MpiRecvInfo),
    Irecv(MpiRecvInfo),
    // Tests
    Test,
    Testall,
    Testany,
    Testsome,
    // Waits
    Wait { request_id: u32 },
    Waitall { requests_ids: Vec<u32> },
    Waitany { request_id: u32 },
    Waitsome { requests_ids: Vec<u32> },
    // Collectives
    Allgather,
    Allgatherv,
    Allreduce,
    Alltoall,
    Alltoallv,
    Alltoallw,
    Barrier,
    Bcast,
    Gather,
    Gatherv,
    Iallgather,
    Iallreduce,
    Ibarrier,
    Ibcast,
    Iagther,
    Iagtherv,
    Ireduce,
    Iscatter,
    Iscatterv,
    Reduce,
    Scatter,
    Scatterv,
    // One-sided
    Accumulate,
    Get,
    Put,
}

pub enum MpiOpKind {
    OneSided,
    PointToPoint,
    Collective,
}

impl MpiOp {
    pub fn get_kind(&self) -> Option<MpiOpKind> {
        match self {
            // Point-to-Point operations
            MpiOp::Send(_) | MpiOp::Isend(_) | MpiOp::Recv(_) | MpiOp::Irecv(_) => {
                Some(MpiOpKind::PointToPoint)
            }
            // Collective operations
            MpiOp::Allgather
            | MpiOp::Allgatherv
            | MpiOp::Allreduce
            | MpiOp::Alltoall
            | MpiOp::Alltoallv
            | MpiOp::Alltoallw
            | MpiOp::Barrier
            | MpiOp::Bcast
            | MpiOp::Gather
            | MpiOp::Gatherv
            | MpiOp::Iallgather
            | MpiOp::Iallreduce
            | MpiOp::Ibarrier
            | MpiOp::Ibcast
            | MpiOp::Iagther
            | MpiOp::Iagtherv
            | MpiOp::Ireduce
            | MpiOp::Iscatter
            | MpiOp::Iscatterv
            | MpiOp::Reduce
            | MpiOp::Scatter
            | MpiOp::Scatterv => Some(MpiOpKind::Collective),
            // One-sided
            MpiOp::Accumulate | MpiOp::Get | MpiOp::Put => Some(MpiOpKind::OneSided),
            // Other operations
            MpiOp::Test
            | MpiOp::Testall
            | MpiOp::Testany
            | MpiOp::Testsome
            | MpiOp::Wait { .. }
            | MpiOp::Waitall { .. }
            | MpiOp::Waitany { .. }
            | MpiOp::Waitsome { .. } => None,
        }
    }
}

#[derive(Default, Clone, Copy, Debug, Encode, Decode)]
pub struct MpiSendInfo {
    pub dest_rank: Option<u32>,
    pub tag: Option<u32>,
    pub comm: MpiComm,
    pub count: u32,
    // TODO: make custom enum for datatypes
    pub dtty: u32,
    // Some(_) for asyncs
    pub request: Option<u32>,
}

#[derive(Default, Clone, Copy, Debug, Encode, Decode)]
pub struct MpiRecvInfo {
    pub src_rank: Option<u32>,
    pub tag: Option<u32>,
    pub comm: MpiComm,
    pub count: u32,
    // TODO: make custom enum for datatypes
    pub dtty: u32,
    // Some(_) for asyncs
    pub request: Option<u32>,
}

#[derive(Default, Clone, Copy, Debug, Encode, Decode)]
pub enum MpiComm {
    #[default]
    World,
    Other(u32),
}

pub trait TraceParser {
    fn parse_trace(&self) -> crate::MpiAppTrace;
}

impl MpiAppTrace {
    pub fn metadata(&self) -> &MpiAppMeta {
        &self.metadata
    }

    pub fn dumps(&self) -> &[(u32, Vec<MpiTrace>)] {
        self.dumps.as_ref()
    }
}
