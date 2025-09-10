import { ChatOllama } from "@langchain/ollama";
import { SerpAPI } from "@langchain/community/tools/serpapi";
import { Calculator } from "@langchain/community/tools/calculator";
import { AgentExecutor, createReactAgent } from "langchain/agents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { config } from 'dotenv';
config({ path: '../.env' });

export const run = async () => {
    const model = new ChatOllama({temperature: 0, model: "llama3"});
    // A tool is a function that performs a specific duty
    // SerpAPI for example accesses google serch resulrs in real time
    const tools = [new Calculator(), new SerpAPI()];
    
    const prompt = ChatPromptTemplate.fromMessages([
        {
            role: "system",
            content: "You are a helpful assistant that can use tools. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}" },
    ]);
    
    const agent = await createReactAgent({
        llm: model,
        tools,
        prompt
    });
    const executor = new AgentExecutor({
        agent,
        tools,
        verbose: true,
        maxIterations: 10
    });
    console.log("Loaded agent.");

    const input = "What are the total number of countries in Asia raised to the power of 3?";
    console.log(`Executing with input "${input}"...`);

    const result = await executor.invoke({ input });
    console.log(`Got output: ${result.output}`);
    /*
        Got output: The result of 15 raised to the power of 3 is 3375.
     */
};

run();