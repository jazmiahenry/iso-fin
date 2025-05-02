import streamlit as st
from iso_financial_mvp.report_generator.disclaimer_template import get_standard_disclaimer

def show_disclaimer(expanded=False):
    """
    Display a collapsible disclaimer component
    
    Args:
        expanded: Whether the disclaimer should be expanded by default
    """
    with st.expander("📋 Legal Disclaimer", expanded=expanded):
        # Get the standard disclaimer text
        disclaimer_text = get_standard_disclaimer()
        
        # Split the disclaimer into paragraphs for better formatting
        paragraphs = disclaimer_text.split('\n\n')
        
        # Display title
        st.subheader("IMPORTANT DISCLAIMER: NOT FINANCIAL ADVICE")
        
        # Display each paragraph
        for paragraph in paragraphs:
            if paragraph.strip():  # Only display non-empty paragraphs
                st.write(paragraph)
        
        # Add timestamp
        st.caption(f"Last updated: April 18, 2025")