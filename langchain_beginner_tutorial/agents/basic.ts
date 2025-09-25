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
            content: 
`You are a helpful assistant that can use tools. You have access to the following tools:

{tools}
tool_names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL: When you get a successful result from a tool, you MUST provide a "Final Answer" and stop. Do not repeat the same action.

IMPORTANT: For calculator, only send simple math expressions like "45*3", not descriptions.

Begin!

Question: {input}
Thought: {agent_scratchpad}`,
            tools: tools.map(tool => tool.name).join(", "),
            tool_names: tools.map(tool => tool.name).join(", ")
        }
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
        maxIterations: 3,
        returnIntermediateSteps: true
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