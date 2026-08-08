"""Tests for amica.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from amica.config import AmicaConfig


def test_default_config_valid():
    """Test that default configuration instantiates without errors."""
    config = AmicaConfig()
    assert config.num_models == 1
    assert config.num_mix_comps == 3
    assert config.lrate == 0.01
    assert config.chunk_size == "auto"
    assert config.outdir is None


def test_validations():
    with pytest.raises(ValueError, match="num_models must be >= 1"):
        AmicaConfig(num_models=0)
    with pytest.raises(ValueError, match="num_models must be >= 1"):
        AmicaConfig(num_models=-1)
    with pytest.raises(ValueError, match="num_mix_comps must be >= 1"):
        AmicaConfig(num_mix_comps=0)
    with pytest.raises(ValueError, match="lrate must be > 0"):
        AmicaConfig(lrate=0.0)
    with pytest.raises(ValueError, match="lrate must be > 0"):
        AmicaConfig(lrate=-0.1)
    with pytest.raises(ValueError, match=r"minrho must be >= 1.0"):
        AmicaConfig(minrho=0.9)
    with pytest.raises(ValueError, match=r"maxrho must be <= 2.0"):
        AmicaConfig(maxrho=2.1)
    with pytest.raises(ValueError, match="minrho must be <= maxrho"):
        AmicaConfig(minrho=1.5, maxrho=1.2)
    with pytest.raises(ValueError, match="max_decs must be >= 0"):
        AmicaConfig(max_decs=-1)
    with pytest.raises(ValueError, match="max_incs must be >= 0"):
        AmicaConfig(max_incs=-1)
    with pytest.raises(ValueError, match="chunk_size must be an int >= 1"):
        AmicaConfig(chunk_size=0)
    # Valid explicit chunk size
    config = AmicaConfig(chunk_size=1024)
    assert config.chunk_size == 1024


def test_outdir_path_conversion():
    config = AmicaConfig(outdir="/tmp/amica")
    assert isinstance(config.outdir, Path)
    assert config.outdir.as_posix() == "/tmp/amica"


def test_invalid_estep():
    import pytest

    from amica import AmicaConfig

    with pytest.raises(ValueError, match="estep must be 'auto', 'fused', or 'classic'"):
        AmicaConfig(estep="invalid")
