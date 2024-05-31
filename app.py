import os
import time
import urllib.parse
import json

import streamlit as st
import templates
import click

from backend import (
    add_url,
    predict,
    setup_db,
    list_glossary,
    list_documents,
    query_database,
    get_query_return_nums,
)

IP = os.environ.get("PDF_FILE_SERVICE_HOST", "localhost")
LOGO_URL = "https://superduperdb-public-demo.s3.amazonaws.com/superduper_logo.png"

def set_session_state():
    """ """
    # default values
    if "search" not in st.session_state:
        st.session_state.search = None
    if "tags" not in st.session_state:
        st.session_state.tags = None
    if "page" not in st.session_state:
        st.session_state.page = 1

    # get parameters in url
    para = st.query_params
    if "search" in para:
        st.session_state.search = urllib.parse.unquote(para["search"][0])
    if "tags" in para:
        st.session_state.tags = para["tags"][0]
    if "page" in para:
        st.session_state.page = int(para["page"][0])


@click.command()
@click.option("--reset", is_flag=True, help="Reset the database.")
def main(reset):
    st.set_page_config(page_title="SuperDuperDB Application", page_icon=":crystal_ball:")
    st.logo(LOGO_URL)
    st.header("SuperDuper App Demo: NVCA Legal Docs - RAG")
    set_session_state()
    db = st.cache_resource(setup_db)(reset=reset)
    st.write(templates.load_css(), unsafe_allow_html=True)
    [
        tab_qa,
        tab_glossary,
        tab_documents,
        tab_query,
    ] = st.tabs(
        [
            "Chat",
            "Glossary",
            "Documents",
            "Query",
        ]
    )
    with tab_qa:
        app_qa(db)
    with tab_glossary:
        app_glossary(db)
    with tab_documents:
        app_documents(db)
    with tab_query:
        app_query(db)


def app_add_url(db):
    print("add url")
    st.subheader("Add datas from S3 Urls")
    url = st.text_input("Enter PDF S3 url:")
    if not url.strip():
        return
    print(url)
    if add_url(db, url):
        st.write("##### Success!")


def app_qa(db):
    """search layout"""
    # load css
    page_size = 20
    if st.session_state.search is None:
        search = st.text_input("Enter search query:")
    else:
        search = st.text_input("Enter search query:", st.session_state.search)

    if search:
        print("search", search)
        # reset tags when receive new search words
        if search != st.session_state.search:
            st.session_state.tags = None
        # reset search word
        st.session_state.search = None
        from_i = (st.session_state.page - 1) * page_size

        start_time = time.time()
        answer, contexts = predict(db, search)
        st.write("##### Response:")
        st.write(answer)
        st.divider()

        cost_time = time.time() - start_time
        total_hits = len(contexts)
        if total_hits > 0:
            # show number of results and time taken
            st.write(
                templates.number_of_results(total_hits, cost_time),
                unsafe_allow_html=True,
            )
            render_contexts(search, contexts, from_i)

            # pagination
            if total_hits > page_size:
                total_pages = (total_hits + page_size - 1) // page_size
                pagination_html = templates.pagination(
                    total_pages,
                    search,
                    st.session_state.page,
                    st.session_state.tags,
                )
                st.write(pagination_html, unsafe_allow_html=True)
        else:
            # no result found
            st.write(templates.no_result_html(), unsafe_allow_html=True)


def render_contexts(search, contexts, from_i=0):
    # search results
    # RESULT
    i = 0

    contexts = sorted(contexts, key=lambda x: x[-1], reverse=True)
    for context in contexts:
        page_number, url, text, dst, score = context
        title = " ".join(text.split(" ")[:10])
        title = title.replace("\n", " ")

        res = {}
        res["url"] = f"http://{IP}:8000/" + dst + f"#page={page_number}"
        res["page_no"] = page_number
        res["highlights"] = text
        res["title"] = title
        score = str(round(score, 2))

        st.write(templates.search_result(i + from_i, **res), unsafe_allow_html=True)
        i += 1

        if i < 5:
            tags = [score, "Used in AI Answer"]
        else:
            tags = [score, "Other Results"]

        tags_html = templates.tag_boxes(search, tags, active_tag=score)
        st.write(tags_html, unsafe_allow_html=True)


def app_glossary(db):
    glossaries = list_glossary(db)
    item2data = {data["item"]: data for data in glossaries}
    item_list = sorted(item2data.keys())
    option = st.selectbox(
        "Choose a term mentioned in NVCA Legal Docs:",
        item_list,
    )

    data = item2data[option]
    st.write("**Definition:**")
    st.write(data["definition"])
    search = f"What is {data['item']}"
    button = st.button("More details")
    if button:
        answer, contexts = predict(db, search)
        st.write(answer)
        render_contexts(search, contexts)


def app_documents(db):
    documents = list_documents(db)
    uris = [doc["uri"] for doc in documents]
    selected_uri = st.selectbox("Choose a document to view:", uris)
    selected_doc = [doc for doc in documents if doc["uri"] == selected_uri][0]
    url = f"http://{IP}:8000/" + selected_doc["base_path"]
    st.write(f"**[View document]({url})**")
    st.markdown("#### Summary")
    st.write(selected_doc["summary"])

    # st.markdown("### Quickly Chat With AI On This Document")
    # search = st.text_input("Input your question:")
    # if search.strip():
    #     answer, contexts = predict(db, search)
    #
    #     st.write("##### Response:")
    #     st.write(answer)
    #
    #     render_contexts(search, contexts)


def reset_selected_index():
    st.session_state["selected_index_str"] = None


def app_query(db):
    collection_names = sorted(db.databackend.db.list_collection_names(), reverse=True)

    selected_collection = st.selectbox(
        "Choose a collection to search in:",
        collection_names,
        on_change=reset_selected_index,
    )

    query_string = st.text_area(
        "Please input query (JSON format)",
        "{}",
        on_change=reset_selected_index,
    )

    query_button = st.button("Query", on_click=reset_selected_index)

    if not (query_button or "selected_index_str" in st.session_state):
        return

    try:
        query_dict = json.loads(query_string)
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please correct it and try again.")
        return

    num = get_query_return_nums(db, selected_collection, query_dict)
    st.write(f"Number of results: {num}")

    if num > 0:
        col1, col2 = st.columns([5, 1])

        result_indices = [str(i + 1) for i in range(min(num, 10))]

        with col2:
            current_selection = st.radio(
                "Select result number",
                result_indices,
                key="selected_index",
            )

        selected_index = int(current_selection) - 1
        st.session_state[
            "selected_index_str"
        ] = current_selection

        datas = query_database(
            db, selected_collection, query_dict, skip=selected_index, limit=1
        )

        with col1:
            if datas:
                st.json(datas[0])
            else:
                st.write("No result found")
    else:
        st.write("No result found")


if __name__ == "__main__":
    main()
