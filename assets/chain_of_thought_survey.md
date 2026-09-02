# A Survey of Chain-of-Thought Reasoning in Large Language Models

Chain-of-Thought (CoT) prompting has emerged as a pivotal technique for unlocking the reasoning capabilities of large language models (LLMs) [2]. By decomposing problems into intermediate steps, CoT has significantly improved performance on complex tasks such as arithmetic, commonsense, and symbolic reasoning [2]. Recent advancements have led to the development of Reasoning Large Language Models (RLLMs) like OpenAI's o1 and DeepSeek-R1, which leverage extended, more deliberate reasoning processes [1]. This survey synthesizes current research to provide a structured overview of CoT, focusing on its theoretical origins, a taxonomy of techniques, the distinction between short and long reasoning chains, core methodologies, efficiency challenges, limitations, and future directions.

## Theoretical Foundations and Cognitive Science Origins

The success of CoT prompting has spurred fundamental research into its underlying mechanisms. A prominent line of inquiry draws inspiration from cognitive neuroscience to build a theoretical framework for understanding how reasoning capability emerges through CoT [2]. Specifically, the **Hopfieldian view of cognition** offers a powerful lens, modeling cognitive processes as movements within low-dimensional representation spaces in the brain [2]. This perspective can be mapped onto the CoT process in LLMs: external **stimuli** (such as the prompt "Let's think step by step" in zero-shot CoT or the sequence of demonstrations in few-shot CoT) trigger the reasoning process, which corresponds to the model's internal computations moving through conceptual representation spaces, ultimately leading to an **action** or final answer [2]. This framework not only deepens understanding but also enables methods for localizing reasoning errors by analyzing when model activations deviate from these stable spaces, paving the way for more robust and interpretable reasoning [2].

This cognitive foundation is closely related to the distinction between fast, intuitive thinking and slow, deliberative thinking. Achieving human-level intelligence is seen as refining the transition from fast, intuitive **System 1** to slower, logical **System 2** reasoning [5]. While foundational LLMs excel at System 1-like fast decisions, RLLMs aim to embody the step-by-step, deliberate analysis characteristic of System 2 [5].

## A Taxonomy of CoT Prompting Techniques and Variants

The core CoT technique engages LLMs in step-by-step reasoning rather than direct answer generation [2]. This foundational approach has branched into a diverse ecosystem of prompting strategies, which can be systematically categorized.

A primary distinction is between **zero-shot** and **few-shot** CoT paradigms [2]. In zero-shot CoT, the model is prompted with a question and a stimulus instruction (e.g., "Let's think step by step") to elicit reasoning without prior examples [2]. In few-shot CoT, the prompt includes several exemplar question-answer pairs with detailed reasoning chains, providing an in-context demonstration of the desired process [2].

Beyond this, a critical conceptual and practical evolution is the shift from **Short Chain-of-Thought (Short CoT)** to **Long Chain-of-Thought (Long CoT)** reasoning [1]. Short CoT is characterized by relatively shallow and efficient logical processing, prioritizing quick pathfinding to a solution [1]. In contrast, Long CoT involves deeper reasoning, extensive exploration of logical structures, and iterative reflective analysis, representing a more thorough and exhaustive form of deliberative System 2 thinking [1]. The key components of Long CoT are deep reasoning, which ensures rigorous logical steps; exploration, which identifies novel pathways; and reflection, which allows for reassessment and error correction [1].

Further variants have been developed to address specific challenges. **Faithful CoT** frameworks translate natural language queries into symbolic reasoning chains to improve reliability [2]. **Latent or Implicit CoT** methods aim to condense the reasoning process into continuous hidden representations or planning tokens, moving away from generating lengthy explicit token sequences to improve efficiency [1][4]. **Self-consistency** and other **parallel scaling** techniques generate multiple reasoning paths concurrently and employ voting or verification to select the best final answer [1][4].

## Methodologies and System Design

The implementation of advanced CoT reasoning, particularly Long CoT, relies on a synthesis of several core technical approaches.

**Refinement and Self-Correction:** A prominent line of work enhances reasoning through iterative self-refinement mechanisms. This includes prompt-based methods where models generate initial outputs and then use self-feedback to iteratively improve performance [1]. Supervised fine-tuning (SFT) approaches allow models to learn error correction processes from advanced LLMs or through frameworks that provide step-by-step natural language feedback [1]. Furthermore, methods integrating Monte Carlo Tree Search (MCTS) parse backtracking and refinement into natural language, enhancing the learning of reasoning processes [1].

**Structured Reasoning and Exploration:** Advanced RLLMs exhibit structured exploratory behavior, formulating hypotheses and pursuing alternative solution paths [5]. This is often facilitated by macro-level planning and micro-level verification actions (e.g., "Wait", "Hold on") that enable meticulous checking [5]. Techniques like latent space reasoning condense or guide the reasoning process within continuous vector spaces, using planning tokens or thought vectors to manage complexity and improve efficiency [1][4]. For instance, methods like Coconut (Chain of Continuous Thought) replace explicit token generation by feeding the model's last hidden state back as input for subsequent reasoning steps, while CCoT (Compressed Chain of Thought) fine-tunes models to produce compressed representations of reasoning chains [4].

**Inference-Time Scaling:** A key method for achieving Long CoT is inference-time scaling, which extends the computational budget for reasoning during generation. This can be achieved through **sequential scaling**, which extends reasoning depth within a single generation or through iterative revisions, and **parallel scaling**, which generates multiple candidate reasoning paths simultaneously for selection [1][4]. However, scaling reasoning length is not universally beneficial and is subject to debates about "overthinking," where excessively long chains may harm performance or introduce unnecessary complexity [1].

## Empirical Performance and Model Comparisons

Empirical evaluations reveal significant performance differences across model families and scales when employing CoT prompting. Comprehensive benchmarks like the Chain-of-Thought Hub show that model scale is clearly correlated with reasoning capabilities [3]. As of recent evaluations, leading proprietary models like GPT-4, Claude, and PaLM-2 demonstrate superior performance on complex reasoning tasks compared to open-source models such as LLaMA and Flan-T5 [3]. This performance gap is particularly evident in tasks requiring multi-step reasoning, such as mathematics (GSM8k, MATH), coding (HumanEval), and broad knowledge (MMLU) [3]. Furthermore, the most capable models often leverage reinforcement learning from human feedback (RLHF), indicating that post-pretraining alignment techniques are crucial for unlocking advanced reasoning [3]. These comparisons highlight that for open-source efforts to catch up, the community must focus on building better base models and exploring effective RLHF [3].

## Integration with External Tools and Knowledge

A critical direction for enhancing CoT reasoning is its integration with external tools and knowledge bases to overcome inherent limitations in model knowledge and timeliness [1]. Two primary approaches are prominent:

**Retrieval-Augmented Generation (RAG):** RAG systems enhance reasoning by dynamically retrieving content from external knowledge bases [4]. Recent agentic RAG systems empower models to autonomously decide when and what to retrieve, demonstrating enhanced planning and problem-solving [4]. For example, some methods train RAG models to perform step-by-step retrieval and reasoning over relevant information before generating a final answer, modulating computational cost by adjusting the length and number of sampled retrieval chains [4]. Frameworks like Stream of Search (SoS) and CoRAG incorporate reflection and exploration to boost search accuracy and address unresolved issues [1].

**Tool and API Integration:** Beyond web search, the scope of tool integration has expanded to include code interpreters, specialized solvers, and other external functionalities [5]. Self-learning frameworks allow models to acquire tool-using skills through methods like hint-based learning, teaching them when and how to employ external tools [5]. This is valuable in specialized domains, such as automating the modeling and solving of Operations Research problems by translating natural language into formal models and executable code [5]. Furthermore, Python tools and other APIs are being integrated into Long CoT frameworks through both prompting and training to perform more effective test-time scaling [1].

## Challenges in Reasoning Efficiency

As RLLMs generate longer and more complex reasoning chains, achieving efficiency becomes a distinct and critical challenge separate from general inference acceleration [4].

**Quantifying Reasoning Utility:** A fundamental obstacle is the difficulty in evaluating the contribution of each intermediate reasoning token to the final answer. This lack of granular metrics makes it hard to determine which parts of a reasoning chain can be compressed or pruned without degrading performance, creating a delicate trade-off between conciseness and correctness [4].

**Efficiency Strategies:** Research addresses these challenges through multiple strategies. Length budgeting methods enforce token budgets per step or for the entire reasoning process [4]. System-switch approaches, inspired by dual-process theory, dynamically alternate between fast, intuitive (System 1) and slow, deliberative (System 2) reasoning based on task complexity [4]. Model-switch methods allocate queries to different models or candidate outputs using lightweight controllers [4]. Parallel search strategies generate multiple outputs concurrently but employ early termination or pruning to manage computational cost [4]. Additionally, SFT methods aim to internalize concise reasoning by training on compressed reasoning chains or using latent space tokens [4].

## Empirical Analysis of Limitations and Failure Modes

Despite its successes, CoT reasoning exhibits several important limitations and failure modes that constrain its reliability and trustworthiness.

**Reasoning Unfaithfulness and Hallucination:** A significant issue is that the generated CoT may not faithfully represent the model's actual internal reasoning process. Studies have shown that models can generate plausible-sounding reasoning steps while internally following a different latent process, or that the reasoning chain itself may contain factual errors and hallucinations [4]. This unfaithfulness introduces uncertainty and makes it difficult to trust the reasoning trace as an explanation for the answer [4].

**Safety and Robustness Vulnerabilities:** The explicit, multi-step nature of reasoning opens new attack vectors. **Chain-of-thought attacks**, such as prompt hijacking, can exploit structural vulnerabilities in the reasoning flow [5]. Furthermore, the safety rate of an LRM's internal thinking process can be lower than that of its final answers, meaning sensitive or harmful content may be generated and exposed during reasoning even if the final output is safe [4].

**Generalization and Exploration Limits:** Early external reasoning approaches, like Tree of Thoughts, were constrained by limited exploration space and poor experience sharing across different reasoning paths, which restricted their effectiveness [5]. While modern RLLMs have improved, ensuring robust generalization of reasoning skills across diverse and unseen tasks remains a persistent challenge [5].

**Knowledge Conflicts:** Reasoning accuracy can be undermined when the model's internal knowledge conflicts with information provided in the external context, creating inconsistencies that are difficult to resolve within the reasoning chain [5].

## Evaluation and Analysis

Evaluating CoT reasoning requires benchmarks that assess both the final outcome and the reasoning process itself.

**Benchmarks:** Evaluation benchmarks are categorized by domain and focus. Outcome benchmarks holistically assess performance on complex tasks in mathematics (e.g., MATH, OlympiadBench), coding (e.g., LiveCodeBench, SWEbench), and commonsense puzzles (e.g., ARC, BIGBench Hard) [1][5]. There is a recognized need for process benchmarks that concentrate on the local view of the Long CoT process or individual capabilities [1].

**Error Analysis and Robustness:** Understanding and improving the robustness of the reasoning process is an active area. The Hopfieldian framework enables methods for localizing reasoning errors by analyzing when model activations deviate from stable conceptual representation spaces [2]. Building on this, the **Representation-of-Thought (RoT)** framework leverages the robustness of these representation spaces to enhance the robustness and interpretability of CoT reasoning [2].

## Future Directions

Several research gaps and promising future directions are identified across the surveyed literature.

**Efficiency and Evaluation:** Future work must develop more efficient evaluation frameworks and proxy tasks that account for the computational cost of long reasoning chains, moving beyond metrics that consider only the final answer [5]. Improving reasoning efficiency requires better integration of model-aware adaptive budgeting, advanced pruning of reasoning traces, and balancing search depth with width [4].

**Advanced Reasoning Paradigms:** Directions include the integration of multi-modal reasoning, combining visual and textual reasoning [1]. Enhancing knowledge frameworks and exploring the synergy between different technical approaches—such as symbolic logic, MCTS, and reinforcement learning—with foundational LLMs remains crucial [5]. Furthermore, managing the phenomena of inference-time scaling and "overthinking" requires deeper understanding to optimize the relationship between reasoning length and accuracy [1].

**Generalization and Robustness:** Developing models that generalize reasoning capabilities across diverse and unseen tasks is a persistent challenge [5]. Enhancing the self-improvement and self-correction capacities of models without oracle feedback is another critical avenue for creating more autonomous and reliable reasoning systems [1]. Addressing trustworthiness challenges—including interpretability, knowledge conflict resolution, and safety hardening—will be essential for real-world deployment [5].

## References

- [1] [2503.09567v5] [Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models (2025)](https://arxiv.org/abs/2503.09567v5)
- [2] [2410.03595v1] [Understanding Reasoning in Chain-of-Thought from the Hopfieldian View (2024)](https://arxiv.org/abs/2410.03595v1)
- [3] [2305.17306v1] [Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance (2023)](https://arxiv.org/abs/2305.17306v1)
- [4] [2503.21614v1] [A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond (2025)](https://arxiv.org/abs/2503.21614v1)
- [5] [2502.17419v6] [From System 1 to System 2: A Survey of Reasoning Large Language Models (2025)](https://arxiv.org/abs/2502.17419v6)

 *Only references that have been used will be listed.*