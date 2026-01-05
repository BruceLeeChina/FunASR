#!/usr/bin/env python3
"""
markdown_to_html.py - 将Markdown转换为HTML
"""

import markdown
import os
import glob
from datetime import datetime

def convert_markdown_to_html(md_file, html_file):
    """转换单个Markdown文件为HTML"""
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 使用扩展
    extensions = [
        'markdown.extensions.extra',
        'markdown.extensions.codehilite',
        'markdown.extensions.toc',
        'markdown.extensions.tables',
        'markdown.extensions.fenced_code'
    ]

    html_content = markdown.markdown(md_content, extensions=extensions)

    # 创建完整HTML页面
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{os.path.basename(md_file)} - TTS API 文档</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            background-color: #fff;
            margin: 0;
            padding: 20px;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 1200px;
            margin: 0 auto;
            padding: 45px;
        }}
        @media (max-width: 767px) {{
            .markdown-body {{
                padding: 15px;
            }}
        }}
        .nav {{
            background-color: #f6f8fa;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 6px;
        }}
        .nav a {{
            margin-right: 15px;
            text-decoration: none;
            color: #0366d6;
        }}
        .nav a:hover {{
            text-decoration: underline;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eaecef;
            color: #6a737d;
            font-size: 14px;
        }}
    </style>
</head>
<body>

    
    <article class="markdown-body">
        {html_content}
    </article>
    
    <div class="footer">
        最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        <a href="https://github.com/your-repo/tts-api">GitHub</a>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>
</html>"""

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"转换完成: {md_file} -> {html_file}")

def convert_all_markdown(directory):
    """转换目录下所有Markdown文件"""
    md_files = glob.glob(os.path.join(directory, "*.md"))

    for md_file in md_files:
        html_file = os.path.splitext(md_file)[0] + '.html'
        convert_markdown_to_html(md_file, html_file)

if __name__ == "__main__":
    # 设置您的文档目录
    docs_dir = ""
    convert_all_markdown(docs_dir)

# pip install markdown pygments
# python markdown_to_html.py
