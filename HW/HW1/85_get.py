import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 重试策略
retry_strategy = Retry(
    total=3,  # 最大重试次数
    status_forcelist=[429, 500, 502, 503, 504],  # 针对这些状态码进行重试
    allowed_methods=["HEAD", "GET", "OPTIONS"],  # 允许重试的HTTP方法
    backoff_factor=1  # 重试间隔时间
)

# 应用到session
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)

# 基础URL格式
base_url = "https://avalon.law.yale.edu/18th_century/fed"

# 初始化存储文章内容的变量
all_articles = ""

# 遍历1到85篇文章的链接
for i in range(1, 86):  # 从1到85
    # 构建完整的URL
    article_url = f"{base_url}{i:02d}.asp"  # 确保编号为两位数，例如fed01.asp, fed02.asp等
    print(f"Processing URL: {article_url}")

    try:
        # 请求文章内容
        article_response = http.get(article_url)
        article_soup = BeautifulSoup(article_response.content, 'html.parser')

        # 获取文章序号 (位于 div class="document-title")
        article_number_tag = article_soup.find('div', class_='document-title')
        article_number = article_number_tag.get_text(separator="\n").strip() if article_number_tag else f"Article {i}"

        # 获取所有可能的标题和作者信息（先从 <h3> 获取，若无则从 <h4> 获取）
        article_headers = article_soup.find_all(['h3', 'h4'])
        article_header = "\n".join([header.get_text(separator="\n").strip() for header in
                                    article_headers]) if article_headers else "Unknown Title / Author"

        # 获取文章正文内容 (所有 <p> 标签中的文本)
        paragraphs = article_soup.find_all('p')
        article_content = []
        for p in paragraphs:
            text = p.get_text().strip()
            # 如果遇到 "PUBLIUS." 则停止获取后面的内容
            if "PUBLIUS." in text:
                article_content.append(text)
                break
            article_content.append(text)

        article_content = "\n\n".join(article_content) if article_content else "No content available for this article."

        # 将序号、标题、作者和正文拼接到一起
        full_article = f"{article_number}\n{article_header}\n\n{article_content}"

        # 添加分隔符和文章内容
        all_articles += full_article + "\n\n---- ARTICLE END ----\n\n"

    except Exception as e:
        # 如果处理某篇文章时出错，打印错误并继续处理其他文章
        print(f"Error processing {article_url}: {e}")
        continue

# 将所有文章写入txt文件
with open("federalist_papers_complete1.txt", "w", encoding='utf-8') as f:
    f.write(all_articles)

print("所有文章已抓取并保存到 federalist_papers_complete1.txt 文件中！")