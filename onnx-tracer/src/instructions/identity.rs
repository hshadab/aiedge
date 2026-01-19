use crate::{
    instructions::{declare_onnx_instr, ElementWise, VirtualInstructionSequence},
    trace_types::{AtlasCycle, ONNXOpcode},
    utils::VirtualSlotCounter,
};

// Identity operation - passes through input unchanged
declare_onnx_instr!(name = Identity);

impl ElementWise for Identity {
    fn exec(x: u64, _y: u64) -> u64 {
        x
    }
}

impl VirtualInstructionSequence for Identity {
    fn virtual_trace(
        cycle: crate::trace_types::ONNXCycle,
        _K: &mut VirtualSlotCounter,
    ) -> Vec<AtlasCycle> {
        debug_assert_eq!(cycle.instr.opcode, ONNXOpcode::Identity);
        vec![cycle.try_into().unwrap()]
    }

    fn sequence_output(x: &[u64], _y: &[u64]) -> Vec<u64> {
        x.iter().map(|&x| Self::exec(x, 0)).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::instructions::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn virtual_sequence_32() {
        jolt_virtual_sequence_test::<Identity>(ONNXOpcode::Identity, 16);
    }
}
