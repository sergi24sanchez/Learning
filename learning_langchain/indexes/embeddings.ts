import {OllamaEmbeddings} from "@langchain/ollama"
import "dotenv/config"

export const run = async () => {
    console.log("=== Using Ollama Embeddings (Completely Local, No API Key) ===");
    
    // Using Ollama embeddings - completely local, no API key required!
    /*  ollama serve
        ollama pull nomic-embed-tex
    */
    const embeddings = new OllamaEmbeddings({
        model: "nomic-embed-text", // Popular open-source embedding model
        baseUrl: "http://localhost:11434", // Default Ollama URL
    });

    /*
        Embed query from the user
    */
    console.log("\n1. Embedding a single query...");
    const res = await embeddings.embedQuery("Hello, world!");
    console.log("Query vector:", res);

    /*
        Embed documents (convert your text data into vectors)
    */
    console.log("\n2. Embedding multiple documents...");
    const documentRes = await embeddings.embedDocuments([
        "Hello, world!",
        "Bye bye",
        "This is a test document",
    ]);
    console.log("Number of documents:", documentRes.length);
    console.log("Each document vector length:", documentRes[0].length);
    console.log("First document - first 5 dimensions:", documentRes[0].slice(0, 5));
};

run();
