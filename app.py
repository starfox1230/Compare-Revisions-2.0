from flask import Flask, render_template_string, request
import difflib
import re
import os
import json
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Existing code...

# Sort case data based on sorting option
def sort_cases(case_data, sort_option):
    if sort_option == "case_number":
        case_data.sort(key=lambda x: int(x['case_num']))
    elif sort_option == "percentage_change":
        case_data.sort(key=lambda x: x['percentage_change'], reverse=True)
    elif sort_option == "summary_score":
        case_data.sort(key=lambda x: x['summary']['score'] if x['summary'] else 0, reverse=True)
    return case_data

@app.route('/', methods=['GET', 'POST'])
def index():
    custom_prompt = request.form.get('custom_prompt', DEFAULT_PROMPT)
    sort_option = request.form.get('sort_option', 'case_number')
    case_data = []

    if request.method == 'POST':
        text_block = request.form['report_text']
        case_data = extract_cases(text_block, custom_prompt)
        case_data = sort_cases(case_data, sort_option)

    template = """
<html>
    <head>
        <title>Radiology Report Diff & Summarizer</title>
        <!-- CSS and other styles... -->
    </head>
    <body>
        <div class="container">
            <h2 class="mt-4">Compare Revisions & Summarize Reports</h2>
            <form method="POST" id="reportForm">
                <!-- Input fields... -->
                <button type="submit" class="btn btn-primary">Compare & Summarize Reports</button>
            </form>
            {% if case_data %}
                <h3>Case Navigation</h3>
                <div class="btn-group" role="group" aria-label="Sort Options">
                    <button type="button" class="btn btn-secondary" onclick="sortCases('case_number')">Sort by Case Number</button>
                    <button type="button" class="btn btn-secondary" onclick="sortCases('percentage_change')">Sort by Percentage Change</button>
                    <button type="button" class="btn btn-secondary" onclick="sortCases('summary_score')">Sort by Summary Score</button>
                </div>
                <ul>
                    {% for case in case_data %}
                        <li>
                            <a href="#case{{ case.case_num }}">Case {{ case.case_num }}</a> - {{ case.percentage_change }}% change - Score: {{ case.summary.score if case.summary else 'N/A' }}
                        </li>
                    {% endfor %}
                </ul>
                <hr>
                {% for case in case_data %}
                    <!-- Case details... -->
                {% endfor %}
            {% endif %}
        </div>
        <script>
            function sortCases(option) {
                const form = document.getElementById('reportForm');
                const sortInput = document.createElement('input');
                sortInput.type = 'hidden';
                sortInput.name = 'sort_option';
                sortInput.value = option;
                form.appendChild(sortInput);
                form.submit();
            }
        </script>
        <!-- Bootstrap JS... -->
    </body>
</html>
    """
    return render_template_string(template, case_data=case_data, custom_prompt=custom_prompt)

if __name__ == '__main__':
    app.run(debug=True)
