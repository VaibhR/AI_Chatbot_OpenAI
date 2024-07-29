from langchain_community.document_loaders import WebBaseLoader


def fetching_data_frm_url(url):
    loader = WebBaseLoader(url)
    return loader.load()


if __name__ == "__main__":
    url = 'https://isamu-website.medium.com/literature-review-on-rag-retrieval-augmented-generation-for-custom' \
          '-domains-325bcef98be4'
    data = fetching_data_frm_url(url)
    for dt in data:
        print(dt)
