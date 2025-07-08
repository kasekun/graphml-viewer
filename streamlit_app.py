#!/usr/bin/env python3
"""
Standalone Streamlit app for GraphML Viewer.
This can be run directly with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now we can import our modules
try:
    from graphml_viewer.interactive import InteractiveViewer

    def main():
        """Main entry point for the Streamlit app."""
        viewer = InteractiveViewer()
        viewer.run()

    if __name__ == "__main__":
        main()

    # For streamlit run
    main()

except Exception as e:
    import streamlit as st

    st.error(f"Error loading GraphML Viewer: {e}")
    st.info("Make sure all dependencies are installed: `uv sync`")
