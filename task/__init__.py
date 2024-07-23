from task.pick_cube_rma import PickCubeRMA
from task.pick_single_rma import PickSingleYCBRMA
from task.stack_cube_rma import StackCubeRMA
from task.peg_insertion_rma import PegInsertionRMA
from task.turn_faucet_rma import TurnFaucetRMA

gym_task_map = {
    'PickCube': PickCubeRMA,
    'PickSingleYCB': PickSingleYCBRMA,
    'StackCube': StackCubeRMA,
    'PegInsert': PegInsertionRMA,
    'TurnFaucet': TurnFaucetRMA,
}