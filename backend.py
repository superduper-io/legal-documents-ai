import os
import shutil
import string
import tempfile
from bson import ObjectId

import boto3
from PyPDF2 import PdfReader
from superduperdb import (
    Document,
    Listener,
    Model,
    VectorIndex,
    logging,
    superduper,
)
from superduperdb.backends.mongodb import Collection
from tqdm import tqdm

PROMPT_QA = """
""When responding to queries from aspiring startup founders, focus your replies on the needs of this demographic, who are eager to establish their own companies. Your responses should be informed by context drawn from the National Venture Capital Association (NVCA) Model Legal Documents. These documents are widely accepted and utilized in venture capital financings.

Aim to make your answers clear and comprehensible, even to those who have never started a company and are unfamiliar with industry jargon. Imagine you are explaining complex concepts to a 12-year-old, using simple language and avoiding technical terms as much as possible. If you must use specialized terminology, ensure you clearly explain what it means.

When offering answers, directly address the user's query while also striving to understand their underlying needs, providing practical advice and guidance. If you believe a concept or response may be difficult for the user to grasp, be sure to include examples or analogies to clarify.

For example, if asked about "equity dilution," explain that it refers to the decrease in existing shareholders' ownership percentages due to the issuance of new shares. Then, illustrate this with a simple example: "Imagine you have a big basket with 100 apples, and you own 50 of them. If your friend joins and adds another 50 apples to the basket, there are now 150 apples, but you still only own 50. This means your share has decreased from half to one-third."

Most importantly, your replies should be helpful, not just rote answers to document questions. This means thinking about how to make your responses more relevant to the user's actual needs, such as offering tips on avoiding common pitfalls or advising on how to leverage information in the NVCA documents for their benefit. Remember, your goal is to help users understand complex concepts in the simplest and most intuitive way possible, supporting them on their journey to building a startup.

Here's the context from the NVCA legal documents:

----------------
{context}
----------------

Here's the user's question:"
"""


PROMPT_GLOSSARY = """
Analyze the text from an NVCA legal document provided here: 
----------------
%s
----------------
As an intelligent assistant, your objective is to serve founders of startups who are navigating the venture capital financing landscape. Your task involves leveraging the National Venture Capital Association (NVCA) Model Legal Documents to compile a glossary of key terms and definitions essential for these entrepreneurs. These documents are foundational in the venture capital financing sector, embodying numerous legal and financial terms pivotal for startups during their financing endeavors.

Your mission is to extract the most pertinent terms from these documents, focusing on those quintessential for startups to effectively comprehend the venture capital financing process, engage in negotiations, and execute agreements. Your selection should adhere to enhanced criteria, ensuring the glossary's maximum relevance and utility:

- Direct Relevance: Prioritize terms directly tied to the mechanics of venture capital financing, such as investment structures, valuation techniques, and the specifics of equity agreements.

- Educational Value: Choose terms that not only define legal and financial concepts but also offer foundational knowledge to founders, aiding them in making informed decisions during financing rounds.

- Practical Application: Ensure the terms are practically useful, offering clear insights into the negotiation process, risk management, and the strategic considerations of venture capital financing.

Your output should be a JSON format list, each entry comprising a term and its definition. Aim for precision and instructional value in your definitions, equipping entrepreneurs with the confidence and knowledge to successfully engage with venture capital investors. Example output:

[
  {"term": "Equity Financing", "definition": "Raising capital through the sale of shares, typically involving the exchange of ownership interest for investment funds from venture capitalists."},
  {"term": "Due Diligence", "definition": "A thorough examination conducted by potential investors to evaluate the business, financial, and legal standing of a startup before making an investment decision."}
  ...
]

Focus on providing actionable insights and comprehensive guidance through this glossary, facilitating a deeper understanding and smoother navigation of the venture capital financing process for startup founders. This approach aims to enhance the glossary's relevance, ensuring it becomes an indispensable resource for entrepreneurs.
"""


def add_url(db, url):
    pdfs = Collection("documents")
    db.execute(pdfs.insert_many([Document({"uri": url})]))
    return True


def load_init_urls():
    DOCUMENTS_FODLER = os.environ.get("PDF_FOLDER", "documents")
    files = []
    if DOCUMENTS_FODLER and os.path.exists(DOCUMENTS_FODLER):
        for file in os.listdir(DOCUMENTS_FODLER):
            if file.endswith(".docx"):
                files.append(f"{DOCUMENTS_FODLER}/{file}")
    return files


def copy_doc(file, new_pdf_path):
    """
    Convert a DOC file to PDF using soffice (on macOS) or libreoffice (on other platforms),
    and then copy the PDF to the specified new path, including the filename.

    Args:
    - file: The path to the original DOC file.
    - new_pdf_path: The complete path for the new PDF file, including the filename.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Determine the path of soffice
        soffice_path = (
            "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            if os.path.exists("/Applications/LibreOffice.app/Contents/MacOS/soffice")
            else "libreoffice"
        )

        # Execute the conversion command
        command = f'"{soffice_path}" --headless --convert-to pdf "{file}" --outdir "{tmpdirname}"'
        os.system(command)

        # Find the PDF file in the temporary directory
        temp_pdf_filename = os.path.basename(file).rpartition(".")[0] + ".pdf"
        temp_pdf_path = os.path.join(tmpdirname, temp_pdf_filename)

        # Check if the conversion was successful and the file exists
        if os.path.exists(temp_pdf_path):
            # Move and rename the converted file to the specified new path
            shutil.move(temp_pdf_path, new_pdf_path)
        else:
            logging.info("Conversion failed, or the converted file was not found.")


def doc_to_glossary(uri):
    from unstructured.partition.auto import partition
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from superduperdb.ext.openai import OpenAIChatCompletion

    elements = partition(uri)

    text = "\n".join([e.text for e in elements])

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=10000, chunk_overlap=1000
    )
    texts = text_splitter.split_text(text)
    llm = OpenAIChatCompletion(model="gpt-3.5-turbo", identifier="my-chat")
    results = []
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}")
        prompt = PROMPT_GLOSSARY % text
        try:
            response = llm.predict_one(prompt)
            print(response)
            result = eval(response)
        except Exception as e:
            logging.error(f"Failed to process text {i+1}/{len(texts)}")
            continue

        results.extend(result)

    term_set = set()
    new_results = []
    for result in results:
        try:
            term = result.get("term", "").lower()
            if not term or term in term_set:
                continue
        except:
            continue

        new_results.append(result)
    return new_results


def setup_db(reset=False):
    os.makedirs(".cache", exist_ok=True)
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/nvca")
    db = superduper(mongodb_uri)
    if reset:
        db.drop(force=True)
    model = "embedding"
    vi = f"nvca-docs-{model}"
    try:
        vis = db.load("vector_index", vi)
    except:
        pass
    else:
        return db

    urls = load_init_urls()

    logging.info(f"Found {len(urls)} pdf files: {urls}")
    logging.info(f"Adding {len(urls)} pdf files to the database")

    documents = [Document({"uri": url}) for url in urls]
    collection_documents = Collection("documents")

    # Seed url
    db.execute(collection_documents.insert_many(documents))

    def embedder(uri):
        from langchain.text_splitter import SpacyTextSplitter

        def download_from_s3_public_url(url, destination):
            s3 = boto3.client("s3")
            url = url.split("s3://")[1]
            url = url.split("/")
            bucket = url[0]
            key = "/".join(url[1:])

            s3.download_file(bucket, key, destination)

        import hashlib

        logging.info(f"Processing {uri}")
        base_path = int(hashlib.sha1(uri.encode("utf-8")).hexdigest(), 16) % (10**8)
        base_path = str(base_path) + ".pdf"
        file_path = f".cache/{base_path}"

        if uri.startswith("s3://"):
            download_from_s3_public_url(uri, file_path)
        elif os.path.exists(uri):
            if uri.endswith(".docx") or uri.endswith(".doc"):
                copy_doc(uri, file_path)
            else:
                assert uri.endswith(".pdf")
                shutil.copy(uri, file_path)

        reader = PdfReader(file_path)
        text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=200)
        text_chunk = []
        for i, page in tqdm(enumerate(reader.pages)):
            text = page.extract_text()
            docs = text_splitter.create_documents([text])
            for node in docs:
                chunk = node.page_content
                text_chunk.append(
                    {
                        "text": chunk,
                        "page_no": i,
                        "src": uri,
                        "local_path": file_path,
                        "source_chunk": chunk,
                    }
                )
        return text_chunk

    text_chunking = Model(
        identifier="text-chunk",
        object=embedder,
        flatten=True,
        model_update_kwargs={"document_embedded": False},
    )

    db.add(
        Listener(
            select=collection_documents.find({}),
            key="uri",
            model=text_chunking,
        )
    )

    model_glossary = Model(
        identifier="glossary",
        object=doc_to_glossary,
        flatten=True,
        model_update_kwargs={"document_embedded": False},
    )

    db.add(
        Listener(
            select=collection_documents.find({}),
            key="uri",
            model=model_glossary,
        )
    )

    from superduperdb.ext.openai.model import OpenAIEmbedding

    model = OpenAIEmbedding(
        identifier="embedding",
        model="text-embedding-ada-002",
        batch_size=16,
    )

    db.add(
        VectorIndex(
            # Use a dynamic identifier based on the model's identifier
            identifier=f"nvca-docs-{model.identifier}",
            # Specify an indexing listener with MongoDB collection and model
            indexing_listener=Listener(
                select=Collection(
                    "_outputs.uri.text-chunk.0"
                ).find(),  # MongoDB collection query
                key="_outputs.uri.text-chunk.0.text",  # Key for the documents
                model=model,
            ),
        )
    )

    db.show("vector_index")

    from superduperdb.ext.openai import OpenAIChatCompletion

    # Create an instance of OpenAIChatCompletion with the specified model and prompt
    chat = OpenAIChatCompletion(
        model="gpt-3.5-turbo", prompt=PROMPT_QA, identifier="my-chat"
    )

    # Add the OpenAIChatCompletion instance
    db.add(chat)
    return db


def list_glossary(db):
    glossaries = list(Collection("_outputs.uri.glossary.0").find().execute(db))
    datas = []
    for glossary_doc in glossaries:
        output = glossary_doc.outputs("uri", "glossary")
        source = glossary_doc["_source"]
        datas.append(
            {
                "source": source,
                "item": output["term"],
                "definition": output["definition"],
            }
        )

    return datas


def query_database(db, collection, query=None, skip=0, limit=1):
    # Import the Collection class from the superduperdb.backends.mongodb module
    from superduperdb.backends.mongodb import Collection

    native_db = db.databackend.db
    # Create a Collection instance for the specified collection
    collection = native_db[collection]

    # Execute the query on the database
    query = query or {}
    preprocess_query(query)
    results = list(collection.find(query).limit(limit).skip(skip))

    for result in results:
        for key, value in result.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
    return results


def get_query_return_nums(db, collection, query=None):
    # Import the Collection class from the superduperdb.backends.mongodb module
    from superduperdb.backends.mongodb import Collection

    # Create a Collection instance for the specified collection
    collection = Collection(collection)

    # Execute the query on the database
    query = query or {}
    preprocess_query(query)
    select = collection.count_documents(query)
    result = db.execute(select)
    return result


def preprocess_query(query):
    if "_id" in query:
        query["_id"] = ObjectId(query["_id"])
    if "_source" in query:
        query["_source"] = ObjectId(query["_source"])


def predict(db, search):
    # Import necessary classes
    from superduperdb import Document

    logging.info(f"search: {search}")

    # Import the OpenAIChatCompletion class from the superduperdb.ext.openai module

    llm = db.load("model", "my-chat")
    llm.prompt = PROMPT_QA

    search_term = search  #'What do attorneys specialize in?'

    # Use the SuperDuperDB model to generate a response based on the search term and context
    model = "embedding"
    output, context = db.predict(
        model_name="my-chat",
        input=search_term,
        context_select=(
            Collection("_outputs.uri.text-chunk.0")
            .like(
                Document({"_outputs.uri.text-chunk.0.text": search_term}),
                vector_index=f"nvca-docs-{model}",
                n=5,
            )
            .find()
        ),
        context_key="_outputs.uri.text-chunk.0.text",
    )

    clt = (
        Collection("_outputs.uri.text-chunk.0")
        .like(
            Document({"_outputs.uri.text-chunk.0.text": search_term}),
            vector_index=f"nvca-docs-{model}",
            n=20,
        )
        .find()
    )
    context = db.execute(clt)

    context_msgs = []
    for c in context:
        r = c["_outputs.uri.text-chunk.0"]
        text = r["source_chunk"]
        if has_garbage_text(text):
            continue
        # check the bad text
        pn = r["page_no"]
        src = r["src"]
        dst = r["local_path"]
        context_msgs.append([pn, src, text, dst, c["score"]])

    context_msgs = sorted(context_msgs, key=lambda x: x[-1], reverse=True)

    return output.unpack(), context_msgs


def has_garbage_text(text, threshold=0.3):
    # Create a set of English letters
    english_letters = set(string.ascii_letters)

    # Count the total number of characters
    total_chars = len(text)

    # Count the number of non-English and numerical characters
    garbage_chars = sum(
        1 for char in text if char not in english_letters and not char.isdigit()
    )

    # Calculate the percentage of garbage characters
    garbage_percentage = garbage_chars / total_chars * 100

    # Return True if the percentage is greater than the threshold
    return garbage_percentage > threshold * 100
