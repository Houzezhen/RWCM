"""Exercise the training loop's checkpoint writes without loading a ViT."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from world_critic.config import TrainConfig


class CheckpointRetentionTest(unittest.TestCase):
    def run_saves(self, last_only):
        tree = ast.parse((Path(__file__).parents[1] / 'world_critic/train.py').read_text(encoding='utf-8'))
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'epoch')
        first = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.BoolOp)
                     and ast.unparse(node.test).startswith('is_best and'))
        statements = loop.body[first:first+3]
        code = compile(ast.Module(body=statements, type_ignores=[]), '<checkpoint writes>', 'exec')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def save(path, model, optimizer, scheduler, config, epoch, step, metric, ctx):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(epoch))
            config = TrainConfig(epochs=2, checkpoint_last_only=last_only)
            env = dict(output_dir=root, config=config, save_training_checkpoint=save,
                       model=None, optimizer=None, scheduler=None, global_step=1,
                       best_metric=0.1, ctx=SimpleNamespace(), is_best=True)
            for epoch in range(2):
                env['epoch'] = epoch
                exec(code, env)
            return {p.name: p.read_text() for p in (root/'checkpoints').iterdir()}

    def test_only_last_is_overwritten(self):
        self.assertEqual(self.run_saves(True), {'last.pt': '1'})

    def test_default_retains_existing_behavior(self):
        self.assertEqual(self.run_saves(False), {'best.pt': '1', 'epoch-0000.pt': '0',
                                               'epoch-0001.pt': '1', 'last.pt': '1'})


if __name__ == '__main__':
    unittest.main()
