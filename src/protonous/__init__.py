"""protonous: a lightweight agent framework over OpenRouter.

Everything an agent is built from, importable from the top level:

    from protonous import Model, OpenrouterClient, OpenrouterConfig, Toolbox, get_settings

Lower-level pieces stay in their modules: `protonous.tools` (by_name, execute),
`protonous.normalizers` (the provider wire format), `protonous.cli`.
"""
from .client import OpenrouterClient, OpenrouterConfig, collect
from .context import (
    Message,
    MessageChunk,
    ReasoningDetail,
    Role,
    Step,
    ToolCall,
    ToolSpec,
    Usage,
    check_pairing,
)
from .errors import (
    ConfigurationError,
    LayerError,
    NotASessionError,
    ProtonousError,
    ProviderError,
    RunExistsError,
    SessionExistsError,
    StreamIncompleteError,
    ToolDefinitionError,
    ToolPairingError,
    TraceFormatError,
)
from .layers import ContextWindow, Layer, SingleMessageLayer, SlidingWindowLayer, StaticLayer
from .model import Model, ReplyStream
from .settings import Settings, get_settings
from .tools import Tool, Toolbox, run_tools, tool
from .trace import (
    ErrorRecord,
    RunError,
    Session,
    SessionTrace,
    Trace,
    Tracer,
    load_session,
    load_trace,
)

__version__ = "0.0.1"

__all__ = [
    "ConfigurationError",
    "ContextWindow",
    "ErrorRecord",
    "Layer",
    "LayerError",
    "Message",
    "MessageChunk",
    "Model",
    "NotASessionError",
    "OpenrouterClient",
    "OpenrouterConfig",
    "ProtonousError",
    "ProviderError",
    "ReasoningDetail",
    "ReplyStream",
    "Role",
    "RunError",
    "RunExistsError",
    "Session",
    "SessionExistsError",
    "SessionTrace",
    "Settings",
    "SingleMessageLayer",
    "SlidingWindowLayer",
    "StaticLayer",
    "Step",
    "StreamIncompleteError",
    "Tool",
    "ToolCall",
    "ToolDefinitionError",
    "ToolPairingError",
    "ToolSpec",
    "Toolbox",
    "Trace",
    "TraceFormatError",
    "Tracer",
    "Usage",
    "check_pairing",
    "collect",
    "get_settings",
    "load_session",
    "load_trace",
    "run_tools",
    "tool",
]
