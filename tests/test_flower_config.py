"""Tests for Flower app configuration and imports."""

import pytest
import tomli
from pathlib import Path


# =============================================================================
# Configuration Tests
# =============================================================================

class TestPyprojectConfig:
    """Test pyproject.toml Flower configuration."""

    @pytest.fixture
    def pyproject_path(self):
        """Path to pyproject.toml."""
        return Path(__file__).parent.parent / "pyproject.toml"

    @pytest.fixture
    def flwr_config(self, pyproject_path):
        """Load Flower configuration from pyproject.toml."""
        with open(pyproject_path, "rb") as f:
            config = tomli.load(f)
        return config.get("tool", {}).get("flwr", {})

    def test_pyproject_exists(self, pyproject_path):
        """Test that pyproject.toml exists."""
        assert pyproject_path.exists(), "pyproject.toml not found"

    def test_has_flwr_config(self, flwr_config):
        """Test that Flower configuration section exists."""
        assert flwr_config, "No [tool.flwr] section in pyproject.toml"

    def test_has_publisher(self, flwr_config):
        """Test that publisher is defined."""
        app_config = flwr_config.get("app", {})
        assert "publisher" in app_config, "Missing publisher in [tool.flwr.app]"

    def test_has_components(self, flwr_config):
        """Test that serverapp and clientapp components are defined."""
        components = flwr_config.get("app", {}).get("components", {})
        assert "serverapp" in components, "Missing serverapp in components"
        assert "clientapp" in components, "Missing clientapp in components"

    def test_components_format(self, flwr_config):
        """Test that component paths follow module:object format."""
        components = flwr_config.get("app", {}).get("components", {})
        
        serverapp = components.get("serverapp", "")
        clientapp = components.get("clientapp", "")
        
        assert ":" in serverapp, "serverapp should be in 'module:object' format"
        assert ":" in clientapp, "clientapp should be in 'module:object' format"

    def test_has_required_config_keys(self, flwr_config):
        """Test that required run config keys exist."""
        run_config = flwr_config.get("app", {}).get("config", {})
        
        required_keys = [
            "num-server-rounds",
            "fraction-train",
            "local-epochs",
            "lr",
            "spam-strategy",
            "spam-alpha",
        ]
        
        for key in required_keys:
            assert key in run_config, f"Missing required config key: {key}"

    def test_config_values_valid(self, flwr_config):
        """Test that config values are within valid ranges."""
        run_config = flwr_config.get("app", {}).get("config", {})
        
        assert run_config.get("num-server-rounds", 0) > 0
        assert 0 < run_config.get("fraction-train", 0) <= 1
        assert run_config.get("local-epochs", 0) > 0
        assert run_config.get("lr", 0) > 0
        assert run_config.get("spam-strategy") in ["iid", "dirichlet"]
        assert run_config.get("spam-alpha", 0) > 0

    def test_has_federation(self, flwr_config):
        """Test that federation configuration exists."""
        federations = flwr_config.get("federations", {})
        assert "default" in federations, "Missing default federation"
        
        default_fed = federations.get("default")
        assert default_fed in federations, f"Default federation '{default_fed}' not defined"

    def test_federation_has_supernodes(self, flwr_config):
        """Test that local-simulation has num-supernodes configured."""
        local_sim = flwr_config.get("federations", {}).get("local-simulation", {})
        options = local_sim.get("options", {})
        
        assert "num-supernodes" in options, "Missing num-supernodes in local-simulation"
        assert options["num-supernodes"] > 0, "num-supernodes must be positive"


# =============================================================================
# App Import Tests
# =============================================================================

class TestFlowerAppImports:
    """Test that Flower apps can be imported correctly."""

    def test_client_app_imports(self):
        """Test that client_app module can be imported."""
        from src import client_app
        assert hasattr(client_app, "app"), "client_app missing 'app' object"

    def test_server_app_imports(self):
        """Test that server_app module can be imported."""
        from src import server_app
        assert hasattr(server_app, "app"), "server_app missing 'app' object"

    def test_client_app_is_correct_type(self):
        """Test that client app is a Flower ClientApp."""
        from src.client_app import app
        from flwr.client import ClientApp
        assert isinstance(app, ClientApp), f"Expected ClientApp, got {type(app)}"

    def test_server_app_is_correct_type(self):
        """Test that server app is a Flower ServerApp."""
        from src.server_app import app
        from flwr.server import ServerApp
        assert isinstance(app, ServerApp), f"Expected ServerApp, got {type(app)}"

    def test_task_module_imports(self):
        """Test that task module can be imported."""
        from src import task
        assert hasattr(task, "distribute_spam"), "task missing 'distribute_spam'"
        assert hasattr(task, "load_data"), "task missing 'load_data'"

    def test_model_module_imports(self):
        """Test that model module can be imported."""
        from src.model import modernbert
        assert hasattr(modernbert, "create_model"), "modernbert missing 'create_model'"
        assert hasattr(modernbert, "get_tokenizer"), "modernbert missing 'get_tokenizer'"
        assert hasattr(modernbert, "load_model"), "modernbert missing 'load_model'"
        assert hasattr(modernbert, "save_model"), "modernbert missing 'save_model'"


# =============================================================================
# Component Path Resolution Tests
# =============================================================================

class TestComponentResolution:
    """Test that component paths in pyproject.toml resolve correctly."""

    def test_serverapp_resolves(self):
        """Test that serverapp component path resolves to correct object."""
        # Parse the component path
        component_path = "src.server_app:app"
        module_path, obj_name = component_path.split(":")
        
        # Import and verify
        import importlib
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)
        
        from flwr.server import ServerApp
        assert isinstance(obj, ServerApp)

    def test_clientapp_resolves(self):
        """Test that clientapp component path resolves to correct object."""
        # Parse the component path
        component_path = "src.client_app:app"
        module_path, obj_name = component_path.split(":")
        
        # Import and verify
        import importlib
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)
        
        from flwr.client import ClientApp
        assert isinstance(obj, ClientApp)
