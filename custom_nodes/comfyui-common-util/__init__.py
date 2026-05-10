from .layer_nodes import NODE_CLASS_MAPPINGS as layer_nodes_class, NODE_DISPLAY_NAME_MAPPINGS as layer_nodes_display
from .light_nodes import NODE_CLASS_MAPPINGS as light_nodes_class, NODE_DISPLAY_NAME_MAPPINGS as light_nodes_display
from .relay_node import NODE_CLASS_MAPPINGS as relay_nodes_class, NODE_DISPLAY_NAME_MAPPINGS as relay_nodes_display


NODE_CLASS_MAPPINGS = {
    **layer_nodes_class,
    **light_nodes_class,
    **relay_nodes_class
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **layer_nodes_display,
    **light_nodes_display,
    **relay_nodes_display
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
