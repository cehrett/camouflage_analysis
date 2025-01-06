# plots.py
# Functions to generate time series plots or other visualizations

from IPython.core.display import display, HTML
from bertopic import BERTopic
import os
from datetime import datetime

def validate_inputs(topic_model, output_path, inline):
    if not isinstance(topic_model, BERTopic):
        raise ValueError("The topic_model must be an instance of BERTopic.")
    
    if not inline and output_path is None:
        output_path = os.getcwd()
    
    return output_path

def generate_html_header():
    return """
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

def get_topics_per_class(topic_model, docs, classes):
    topics_per_class = topic_model.topics_per_class(docs, classes=classes)
    return topics_per_class

def generate_visualizations(topic_model, docs, timestamps, nr_bins, topics_per_class = None):
    visualizations = [
        ("Bar Chart", lambda: topic_model.visualize_barchart().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Heatmap", lambda: topic_model.visualize_heatmap().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Hierarchy", lambda: topic_model.visualize_hierarchy().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Topics", lambda: topic_model.visualize_topics().to_html(full_html=False, include_plotlyjs='cdn')),
        ("Topics Over Time", lambda: topic_model.visualize_topics_over_time(topic_model.topics_over_time(docs, timestamps, nr_bins=nr_bins)).to_html(full_html=False, include_plotlyjs='cdn')),
    ]
    
    if topics_per_class is not None and not topics_per_class.empty:
        visualizations.append((
            "Topics Per Class",
            lambda: topic_model.visualize_topics_per_class(topics_per_class).to_html(full_html=False, include_plotlyjs='cdn')
        ))
    
    return visualizations

def embed_visualizations(html_content, visualizations):
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
    
    return html_content

def generate_html_footer():
    return """
    </body>
    </html>
    """

def save_or_display_report(html_content, output_path, inline):
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

def generate_html_report(topic_model, output_path=None, classes=None, inline=False, docs=None, timestamps=None, nr_bins=80):
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
    output_path = validate_inputs(topic_model, output_path, inline)
    
    html_content = generate_html_header()

    topics_per_class = None
    if classes:
        topics_per_class = get_topics_per_class(topic_model, docs, classes)
    
    visualizations = generate_visualizations(topic_model, docs, timestamps, nr_bins, topics_per_class)
    html_content = embed_visualizations(html_content, visualizations)
    html_content += generate_html_footer()
    
    return save_or_display_report(html_content, output_path, inline)
