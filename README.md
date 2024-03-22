# legal-documents-ai
Demo of SuperDuperDB as a tool and platform for performing legal documents AI, RAG and comprehension
### Installation
- `pip install -r requirements.txt`
- `python3 -m spacy download en`

#### Specific format file dependencies

**doc/docx**
If we want to process files in doc/docx format, we need to install the following dependencies

Linux: 
```
sudo apt update
sudo apt install libreoffice
```

MacOs: 

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask libreoffice
```

### Initilization

#### Start Mongodb service

```
docker run -itd --name mongo -p 27017:27017 mongo
```
#### Create PDF File System Service

1. Create new window and goto root run `python -m http.server 8000`


#### Setup DB And Start streamlit App Service

1. Set OpenAI Key: `export OPENAI_API_KEY=sk-....`
2. Set `export PDF_FILE_SERVICE_HOST=localhost`
3. Set the pdf directory of the initialization data, if necessary: `export DOCUMENTS_FOLDER=documents`, default load the `./documents`
4. `streamlit run app.py`, if you want to reset rebuild the database, you can add `reset`: `streamlit run app.py -- --reset`

