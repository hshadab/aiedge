use crate::jolt::{
    executor::instructions::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};
use onnx_tracer::instructions::identity::Identity;

// Identity is a passthrough operation - handled without lookup tables
impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for Identity {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        None
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for Identity {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (self.0, self.1 as i64)
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn to_lookup_index(&self) -> u64 {
        self.0  // Identity just returns the input
    }

    fn to_lookup_output(&self) -> u64 {
        self.0  // Identity just returns the input unchanged
    }
}
