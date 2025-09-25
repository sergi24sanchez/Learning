import { ChatOllama } from "@langchain/ollama";

export const run = async () => {
    // temperature controls how random/creative the response is. It ranges from 0(deterministic) to 1(max creativity)
    const model = new ChatOllama({temperature: 0.1, model: "llama3"});
    const res = await model.invoke("What is the capital city of France?");
    console.log(res.content);
};

run().catch((error) => {
    console.error("Error running ChatOllama:", error);
});