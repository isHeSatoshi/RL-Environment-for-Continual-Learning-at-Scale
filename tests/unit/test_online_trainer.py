"""
Unit tests for OnlineTrainer, NoveltyGate, ContextPruner, and DPOLearner.
"""

import os
import tempfile
import pytest
from open_continual_env.training.online_trainer import OnlineTrainer
from open_continual_env.training.dpo_learner import DPOLearner
from open_continual_env.utils.novelty_gate import NoveltyGate
from open_continual_env.utils.context_pruner import ContextPruner
from open_continual_env.trajectory.schema import Trajectory
from open_continual_env.memory.faiss_memory import FAISSMemory


def test_novelty_gate():
    memory = FAISSMemory()
    gate = NoveltyGate(memory=memory)

    s1 = gate.compute_novelty("Write binary search")
    assert s1 == 1.0

    t1 = Trajectory(trajectory_id="t1", prompt="Write binary search", generated_code="def search(): pass")
    memory.add(t1)

    s2 = gate.compute_novelty("Write binary search")
    assert s2 < s1


def test_context_pruner():
    trajs = [
        Trajectory(trajectory_id=f"t{i}", prompt=f"Prompt {i}", generated_code=f"def fn_{i}(): pass", reward=1.0)
        for i in range(10)
    ]
    pruned = ContextPruner.prune_retrieved_experiences(trajs, max_tokens=100)
    assert "Reference Example #1" in pruned
    assert ContextPruner.estimate_tokens(pruned) <= 120


def test_online_trainer_queue():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = OnlineTrainer(adapter_dir=tmpdir)
        traj = Trajectory(trajectory_id="t1", prompt="Sort array", generated_code="def sort(): pass", reward=1.0)

        info = trainer.queue("cluster_algorithms", traj, min_batch_size=1)
        assert info["cluster_id"] == "cluster_algorithms"
        assert info["status"] in ("queued", "submitting")
        # Drain the async worker before tmpdir teardown: it writes adapter
        # files into tmpdir, and on Windows rmtree races open handles
        # (WinError 145) under disk load.
        trainer.executor.shutdown(wait=True)


def test_dpo_learner():
    dpo = DPOLearner(buffer_size=10)
    dpo.add_preference(
        prompt="Write add",
        winning_code="def add(a, b): return a + b",
        losing_code="def add(a, b): return 0",
        winning_reward=1.0,
        losing_reward=0.0,
        cluster_id="cluster_math"
    )

    prefs = dpo.get_cluster_preferences("cluster_math")
    assert len(prefs) == 1

    res = dpo.train_step("cluster_math")
    assert res["status"] in ("completed", "simulated")
