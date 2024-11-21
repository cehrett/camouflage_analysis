# plots.py
# Functions to generate time series plots or other visualizations

from IPython.core.display import display, HTML
from bertopic import BERTopic
import os
from datetime import datetime

def generate_html_report(topic_model, output_path=None, class_data=None, inline=False, docs=None, timestamps=None, nr_bins=80):
    """
    Generates an HTML report for a BERTopic model and either saves it to a file or displays it inline in a Jupyter notebook.
    
    Parameters:
        topic_model (BERTopic): A fitted BERTopic instance.
        output_path (str, optional): The directory path to save the HTML report. If None and `inline` is False, defaults to the current working directory.
        class_data (dict, optional): Data required for `visualize_topics_per_class` visualization.
                                     Should be a dictionary with keys:
                                     - "topics_per_class" (DataFrame): DataFrame from topic_model.topics_per_class()
        inline (bool): Whether to display the report inline in a Jupyter notebook. If False, saves to a file.
    
    Returns:
        str or None: Path to the generated HTML report if saved to a file, otherwise None.
    """
    if not isinstance(topic_model, BERTopic):
        raise ValueError("The topic_model must be an instance of BERTopic.")
    
    if not inline and output_path is None:
        output_path = os.getcwd()
    
    # Prepare the HTML content
    html_content = """
    <html>
    <head>
        <title>Topic Modeling Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .visualization { margin-bottom: 40px; }
        </style>
    </head>
    <body>
        <h1>Topic Modeling Report</h1>
    """
    
    # Visualize and add each plot to the report
    visualizations = [
        ("Bar Chart", lambda: topic_model.visualize_barchart().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Heatmap", lambda: topic_model.visualize_heatmap().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Hierarchy", lambda: topic_model.visualize_hierarchy().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Topics", lambda: topic_model.visualize_topics().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Topics Over Time", lambda: topic_model.visualize_topics_over_time(topic_model.topics_over_time(docs, timestamps, nr_bins=nr_bins)).to_html(full_html=False, include_plotlyjs='cdn')),
    ]
    
    if class_data:
        visualizations.append((
            "Topics Per Class",
            lambda: topic_model.visualize_topics_per_class(class_data["topics_per_class"]).to_html(full_html=False, include_plotlyjs='cdn')
        ))
    
    # Embed each visualization into the HTML report
    for title, visualize_func in visualizations:
        try:
            visualization_html = visualize_func()
            html_content += f"""
            <div class="visualization">
                <h2>{title}</h2>
                {visualization_html}
            </div>
            """
        except Exception as e:
            html_content += f"<h2>{title}</h2><p>Could not generate visualization: {str(e)}</p>"
    
    # End the HTML content
    html_content += """
    </body>
    </html>
    """
    
    # Display inline or save to file
    if inline:
        display(HTML(html_content))
        return None
    else:
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_path, f"bertopic_report_{timestamp}.html")
        with open(report_file, "w", encoding="utf-8") as file:
            file.write(html_content)
        return report_file
