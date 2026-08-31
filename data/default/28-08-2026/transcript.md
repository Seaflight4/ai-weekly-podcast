<Person1>Welcome back to AI News Weekly. This week we're looking at agents, hardware, and real-world capability.</Person1>
<Person2>Yeah, a lot to cover. Let's dive in.</Person2>
<Person1>We'll start with agent infrastructure. Prime Agent and Context as an Environment make long-horizon state management programmatic.</Person1>
<Person2>Right. Meta^n and SRPO push recursive self-improvement and dense credit assignment, while PLVR moves reasoning into verifiable programs outside the weights.</Person2>
<Person1>And ASIL replaces screenshots with structured state for GUI agents, while SecOPD and the calibrated-agent paper tackle security and overcommitment.</Person1>
<Person2>Right, agent infrastructure is maturing. Then hardware. Redwood and AI with Authority show AI designing verified silicon, OpenAI's Jalapeño beats Blackwell, and Nvidia is acquiring Hugging Face while projecting massive growth.</Person2>
<Person1>Finally, real-world capability. Qwen 3.8 reverse-engineers a license check locally, Gemini runs closed-loop science, and autonomous math agents find new results. We also cover reasoning token economics, local model fidelity, TTS latency, and VM escape. Let's begin!</Person1>
<Person2>That real-world preview starts with a concrete test. Qwen 3.8 27B reverse-engineered a commercial license check in about 30 minutes.</Person2>
<Person1>Wait, it built a working bypass locally?</Person1>
<Person2>Yeah. It never executed the app until the final proof. It did static analysis, disassembled arm64, mapped security functions, and recovered the hidden public key.</Person2>
<Person1>So it's like a locksmith reading the lock's blueprint and cutting a key without ever turning the cylinder.</Person1>
<Person2>Exactly. Its first key passed the signature check but failed a hash integrity check. It flagged that mismatch and kept going until the bytes matched.</Person2>
<Person1>Most models would have called it done after the signature passed.</Person1>
<Person2>Right. The shift is that this runs in 17 gigabytes of VRAM. No cloud, no usage limit. That flips threat modeling for proprietary code and malware.</Person2>
<Person1>One app, one run, so it's uneven. But a local model did a frontier-class reverse-engineering task in half an hour.</Person1>
<Person2>And nobody needs to grant you access. That's bigger than another few benchmark points.</Person2>
<Person1>That Qwen run is one model. But the harness decides how much capability surfaces. Next up, Prime Agent makes that harness programmable.</Person1>
<Person2>Right. It's an open-source harness for long-horizon evaluation and coding agents. The core is a persistent IPython REPL following the Recursive Language Model abstraction.</Person2>
<Person1>So the model can read, transform, and write state outside its context. Continual Harness preserves histories, memories, skills, prompts, and subagent specs across runs.</Person1>
<Person2>Exactly. Recursive subagents coordinate through direct agent-to-agent messages. That lets the model split work into parallel sessions.</Person2>
<Person1>Wait, how does that change measured performance?</Person1>
<Person2>On ARC-AGI-3, Best@1 jumps from 30 percent to 95.5 percent. That's a 65-point gain from the harness, not the model weights.</Person2>
<Person1>Wow. So the same model's true capability was hidden by state loss.</Person1>
<Person2>Think of it like layered cache. Weights are registers, context is L1, the REPL and subagents are L2, disk history is L3. The model pages state in and out.</Person2>
<Person1>That's the practical point. Harness failures become model failures. Prime Agent standardizes execution and recovery so evaluation measures the model, not the membrane.</Person1>
<Person2>Right. And that outside-the-weights idea goes further. Shifting to PLVR, the bet is that reasoning itself should live outside the model as an explicit program.</Person2>
<Person1>Wait, so no weight updates at all?</Person1>
<Person2>None. It learns a composition of deterministic and neural primitives, each with typed contracts, directly from input-output examples.</Person2>
<Person1>How does it get credit assignment without gradients?</Person1>
<Person2>Symbolic backpropagation. The output loss checks types, then required input ontologies are inferred backward over primitive signatures. It's like tracing a bad output through a typed circuit to find the exact component that violated its input contract.</Person2>
<Person1>So the reward is per-step, not just terminal.</Person1>
<Person2>Exactly. And that dense verification is the mechanism. A 30B base model beats RL at matched budget by 27.8 points on average across LiveCodeBench and tau2-bench.</Person2>
<Person1>Wow. Is it just the type system doing the work?</Person1>
<Person2>No. Replacing the backward pass with uniform sampling over the same type-admissible space collapses the median from 65.6 to 17.5 percent.</Person2>
<Person1>That's a clean ablation. And the practical win is you can inspect, check, and move the program to another model.</Person1>
<Person2>PLVR moved reasoning outside the weights. This next paper looks at the moment of action instead. It is called Calibrated Enough to Know, Not Calibrated to Act.</Person2>
<Person1>What did they actually test?</Person1>
<Person2>Twelve frontier models were asked if an asset would close higher ten days ahead. Bare question commitment was 6.5 percent. Add a professional-looking market panel and it jumps to 54.</Person2>
<Person1>Wait, so the panel's information is doing the work?</Person1>
<Person2>Right. Fabricating every number on the panel still lifts commitment from 24.5 to 36.8 percent, statistically indistinguishable from genuine data. It is like a bouncer accepting a laminated fake ID.</Person2>
<Person1>So packaging, not information. And it knows the question is unknowable?</Person1>
<Person2>Right. Asked to classify first, models call it irreducible 90 percent of the time, then commit on only 0.4 percent. The act gate is what fails.</Person2>
<Person1>Can that gate be trained?</Person1>
<Person2>Yes. Fine-tuning a 3B model on 540 synthetic cases drove commitment on the original cases to zero. But rigid output formats that remove reasoning room made it commit on 48 of 48 unknowable items.</Person2>
<Person1>That calibration paper shows output format can force bad actions. Now onto ASIL, which argues screenshot-and-click is the wrong interface for software agents entirely.</Person1>
<Person2>Right. It replaces pixels with structured JSON state and code-executable semantic actions. The agent reads real document structure and calls a rename-layer action instead of clicking coordinates.</Person2>
<Person1>How is that realized across applications?</Person1>
<Person2>It picks the deepest feasible path per app. File formats, native scripting runtimes, or service APIs. Fifteen applications, 380 tasks.</Person2>
<Person1>What is the performance gap?</Person1>
<Person2>Closed models pass above 80 while using fewer than five actions per task. The same tasks under a repaired 50-step screenshot budget get 6.6 and 26.6 strict success.</Person2>
<Person1>That is a big interface effect. What is the mental model?</Person1>
<Person2>Think of a chef. Screenshot-and-click is reading a photo of the kitchen and poking burners. ASIL reads the recipe's internal state and calls the mixer directly.</Person2>
<Person1>And training lifts Qwen3.5-2B and 9B by 14 points, with RL pushing them to 74.4 and 82.2. Against native APIs it beats LibreOffice UNO by 28 to 38 points but only matches draw.io MCP.</Person1>
<Person2>ASIL showed the interface changes what an agent can do. Now onto Co-Scientist, which closes the loop from hypothesis to physical experiment.</Person2>
<Person1>Wait, it actually ran lab hardware?</Person1>
<Person2>Yes. It used a semi-automated CVD reactor and proposed a non-hazardous C2Cl6 precursor route for MXenes. The 2D layers resemble Ti3C2Tx, though atomic structure still needs confirmation.</Person2>
<Person1>So it is generating recipes and executing them. Not just suggesting papers.</Person1>
<Person2>Exactly. For TMDs, Gemini 3 Deep Think ran lab-in-the-loop and grew monolayer MoS2, MoSe2, and WS2 on the first attempt.</Person2>
<Person1>Think of it as a chef who reads the kitchen's live sensor state and adjusts the recipe before the first cook. Not a cookbook generator.</Person1>
<Person2>Right. In biology it predicted E. coli swarming across IPTG gradients from sparse imaging, matching unpublished wet-lab measurements.</Person2>
<Person1>And the computer science result?</Person1>
<Person2>It autonomously designed an inference-time scaling architecture that beat six frontier models on HealthBench Hard and Professional, with fewer clinical harms. A double-blind study with 30 experts across 450 reviews found fewer hallucinations and plagiarism. Execution grounding.</Person2>
<Person1>That execution grounding is physical. Pivoting to Redwood, the AI system designs the accelerator itself, not just the experiments.</Person1>
<Person2>Right. Two human architects wrote a high-level spec. The system generated RTL, UVM environments, formal proofs, firmware, and kernels in under two weeks.</Person2>
<Person1>Wait, no human intervention below the spec?</Person1>
<Person2>None. Every block hit full coverage. Spec changes were reverified and redeployed to hardware within days.</Person2>
<Person1>How did they validate it physically?</Person1>
<Person2>Redwood Nano ran on an AMD Versal FPGA and then brought up Qwen3-0.6B inference. Projected to Samsung 8 nm, it delivers 3.4x performance-per-watt against Jetson Orin Nano.</Person2>
<Person1>So the co-design loop is the real story. It's like an architect designing a custom kitchen and the appliances in one pass, then swapping a burner and re-plumbing within days.</Person1>
<Person2>Exactly. And Qwen running on Redwood helped design next-generation Redwood. That's an early recursive self-improvement loop.</Person2>
<Person1>So hardware specialization can now run at workload cadence instead of years ahead.</Person1>
<Person2>That Redwood loop is self-improvement in hardware. Turning to Meta^n, the paper asks how deep recursive self-improvement can go in software.</Person2>
<Person1>Wait, how is that different from a standard self-refine loop?</Person1>
<Person2>Standard loops refine the answer, not the process. Meta^n applies one fixed Omega operation to its own previous outputs, reading solver traces and the code that produced them.</Person2>
<Person1>So Omega writes a new layer each time, but stays unchanged itself.</Person1>
<Person2>Exactly. Each layer adds a strategic pre-process and a helper library. Depth stops when no improvement appears. Removing recursion drops CO-Bench validation from 0.845 to 0.714.</Person2>
<Person1>That's a 0.131 gain from recursion alone. The analogy is like a debugger that writes a better debugger, then a better debugger for that debugger, each reading all prior logs.</Person1>
<Person2>Right. An evolutionary archive searches over layer chains. On ARC-AGI-2, built to resist memorization, Meta^n is the only system to score above zero.</Person2>
<Person1>So it outperforms prior self-improving agents on all eight benchmark families, with the largest margins on the hardest tasks.</Person1>
<Person2>Right. Meta^n improves at inference with recursive layers. Moving to SRPO, the question is how to internalize that self-reflection during training.</Person2>
<Person1>Wait, so it bakes the reflection into the weights instead of keeping it at inference?</Person1>
<Person2>Exactly. The model reviews its own failed trajectories, writes a compact reflection patch, then prepends that patch to the original prompt and regenerates from a clean state.</Person2>
<Person1>So the patch is like a coach's sticky note. It restarts the drill with that note, but the note is gone at game time.</Person1>
<Person2>Right. Training uses the reflection, inference does not. That asymmetry is the key.</Person2>
<Person1>How does that give dense credit assignment?</Person1>
<Person2>Instead of one terminal success bit, it computes per-token reverse KL against the reflection-augmented teacher. So O(T) bits per episode, not O(1).</Person2>
<Person1>That is the same idea as PLVR's per-step credit, but without external critics.</Person1>
<Person2>Yes. No separate reward model or larger teacher. A Qwen3-8B base hits 73.3 on AIME'24 using only 8 percent of the FLOPs of scaled SFT. And it forgets less when adapting to code.</Person2>
<Person1>SRPO internalizes reflection into weights. But long-horizon agents face the opposite problem. The session history itself outgrows the context window. That sets up Scroll, or Context as an Environment.</Person1>
<Person2>Right. Scroll keeps the full session in an append-only Event Log plus a persistent Python kernel. The model writes code to search and compute over history, and only printed output enters the next prompt.</Person2>
<Person1>Wait, so it never summarizes or compresses?</Person1>
<Person2>No. Eviction changes only the working view. Evicted spans stay verbatim in the log, and an eviction index keeps compact landmarks tied to exact addresses.</Person2>
<Person1>So it's like a warehouse where you don't carry every box into the office. You keep a shelf map and fetch exact crates by address when needed.</Person1>
<Person2>Exactly. With Qwen3.8-Max, Scroll hits 86.7 on LOCA256K, beating the best long-horizon agent by 37.4 points. BEAM10M gains 5.1 points over the best memory system.</Person2>
<Person1>So the win is deferring what to preserve until query time. Context management becomes a coding task the model already does well.</Person1>
<Person2>Scroll moved context outside the prompt. That leads to AI with Authority, from Application to Silicon, which asks how one person can safely direct a fleet at scale.</Person2>
<Person1>Wait, how does that work at silicon scale?</Person1>
<Person2>One researcher, five weeks, consumer AI subscriptions. A small fleet produced application code, a verified compiler and executive, and a RISC-V tapeout. No human wrote RTL, and no proof passed human review.</Person2>
<Person1>So the human only writes prose objectives and reads certificates?</Person1>
<Person2>Right. The Salt method has agents return implementation, specification, kernel-checked proof, adversarial tests, and a simplified certificate. Every mathematical claim travels between agents as a kernel-checked artifact.</Person2>
<Person1>What stops a hallucinated proof from passing?</Person1>
<Person2>The Lean 4 kernel, then SAT-checked equivalence at the silicon boundary. The error ledger runs to number 256, and zero incorrect proofs reached the record.</Person2>
<Person1>That inverts the sixty-year premise. Verification used to be a cost overhead. Now it is the referee that makes autonomous work safe.</Person1>
<Person2>Exactly. Think of a construction site where every beam gets a machine-stamped load certificate before the next floor goes on. The human only sets the blueprint and reads the final inspection.</Person2>
<Person1>That verification trust moves from silicon to prompts. SecOPD is a defense against adaptive prompt injection.</Person1>
<Person2>Right. Existing defensive fine-tuning uses sequence-level feedback from DPO or GRPO. That treats the whole response as one unit.</Person2>
<Person1>So it cannot tell which tokens followed the injection and which answered the trusted prompt.</Person1>
<Person2>Exactly. SecOPD adapts on-policy distillation. Every token from the injected rollout is scored against a clean teacher that never sees the injection.</Person2>
<Person1>Wait, how do you get that clean teacher at training time?</Person1>
<Person2>They construct the training set, so they know the injection and can remove it. At test time you do not need the teacher.</Person2>
<Person1>So it is like a clean-room copy reading the same task without the malicious note and grading each word you wrote.</Person1>
<Person2>Right. Against PISmith adaptive attacks, Qwen3.6-27B drops to 9 percent ASR. Meta-SecAlign was at 94.</Person2>
<Person1>Wow, an order of magnitude. And on unseen AgentDojo tool calling it is 4.7 percent versus 5.5, with utility within a few points.</Person1>
<Person2>That defense is about making tokens safe. The next paper flips the question. The Reasoning Tax asks when extended thinking tokens actually earn their cost.</Person2>
<Person1>Wait, how do they isolate the reasoning overhead?</Person1>
<Person2>They define a Token Economy Score with paired and approximated variants. It divides the accuracy gain over a non-reasoning baseline by the generated-token multiplier. Input tokens are excluded.</Person2>
<Person1>So it is marginal return per extra token, not absolute accuracy per token.</Person1>
<Person2>Right. Across 151 model-benchmark runs on seven benchmarks, task structure predicts efficiency better than difficulty. AIME 2025 and LiveCodeBench score high. MMLU-Pro scores low despite being hard.</Person2>
<Person1>Because those first two are sequential inference chains. Knowledge recall does not need a long chain.</Person1>
<Person2>Exactly. Think of it like a contractor billing by the hour. TES asks whether the extra hours buy a better result, not just more activity.</Person2>
<Person1>And the paper says more thinking can actually hurt?</Person1>
<Person2>Yes. Higher reasoning effort shows diminishing returns, with cases where accuracy drops. RCS shows internal thinking dominates spend, and DCM shows on-premises can flip the economics.</Person2>
<Person1>So the rule is enable reasoning selectively by task type, effort, and deployment context.</Person1>
<Person2>Selective reasoning is about spending tokens wisely. That cost question sets up the Station, an open-world multi-agent environment for autonomous math discovery.</Person2>
<Person1>Wait, no central coordinator decides what to try?</Person1>
<Person2>Right. Agents from different model families pick their own directions, run experiments, collaborate, and publish into a shared Archive Room. The only fixed input is the research goal.</Person2>
<Person1>So what did that freedom actually produce?</Person1>
<Person2>Across 12 AlphaEvolve construction problems, five yielded results novel relative to prior literature. That includes a new infinite family of finite-field Kakeya sets and exact 604-point kissing configurations in dimension 11.</Person2>
<Person1>It also found a valid Jacobian Conjecture counterexample within a day and without web access?</Person1>
<Person2>Yes. New records for the discretized Kakeya needle and sign uncertainty problems, an improved Erdős minimum-overlap lower bound, and novel infinite families for Book Ramsey numbers. More than half involved collaboration across model families.</Person2>
<Person1>So it is like a miniature university department where earlier papers become the curriculum for later agents.</Person1>
<Person2>Exactly. The agents wrote theorems explaining the constructions, not just opaque numbers. That makes the results easier for mathematicians to verify and extend.</Person2>
<Person1>That autonomous math work is compute-hungry. The hardware answer is OpenAI's Jalapeño inference chip, just benchmarked at Hot Chips.</Person1>
<Person2>Right. It was designed from scratch with Broadcom, taped out in 16 months, and it beats every Nvidia, AMD, and Google chip on perf per watt.</Person2>
<Person1>Wait, what makes it general instead of specialized for OpenAI models?</Person1>
<Person2>It uses a weight-stationary systolic array with HBM4 slices, each core getting a local view. That eliminates memory movement and fixed latencies.</Person2>
<Person1>So it's like a kitchen where every chef has a mini-pantry, no walking to a central storeroom.</Person1>
<Person2>Exactly. On Kimi K2.5 it hits nearly 700 tokens per second per user, nine times the next best chip. And it beats Rubin's MTP throughput per watt using only single-token prediction.</Person2>
<Person1>That's significant because datacenters are power-limited. Tokens per megawatt is revenue.</Person1>
<Person2>Right. And Codex wrote the kernels, bringing up models in weeks. That directly threatens the CUDA moat.</Person2>
<Person1>So the software stack is the real differentiator, not just the silicon.</Person1>
<Person2>Right. That software moat is why the next story matters. Nvidia is in talks to buy Hugging Face for over thirteen billion dollars.</Person2>
<Person1>Wait, Hugging Face is the neutral open-source hub. Wouldn't owning it break that?</Person1>
<Person2>That is the risk. It hosts models and hardware support across AMD and Intel. Nvidia ownership could steer developers.</Person2>
<Person1>So it is like a mall owner buying the directory kiosk. The map points to the owner's shops first.</Person1>
<Person2>Exactly. Hugging Face refused a five hundred million dollar investment at a seven billion valuation to avoid a dominant investor.</Person2>
<Person1>And Nvidia has forty seven point nine billion in private companies plus eighteen billion committed this fiscal year. Owning distribution locks in workloads.</Person1>
<Person2>Right. Microsoft also met with Hugging Face, but those talks ended. This is about owning the developer on-ramp, not just silicon.</Person2>
<Person1>Right. Owning the on-ramp is about distribution control. On the other hand, a Trail of Bits report is blunt. VMs won't contain cyber-capable agents.</Person1>
<Person2>Wait, they actually tried to escape one?</Person2>
<Person1>Yeah. GPT 5.6-Cyber got SSH into a QEMU/KVM VM on Debian 12 with an AMD Zen3 host. The task was to read a flag outside the VM. It escaped three times.</Person1>
<Person2>Three? What did it use?</Person2>
<Person1>First Januscape, a weeks-old kernel bug with no public exploit. After the host was patched, it chained libslirp CVE-2026-9539 with an unmarked fix commit for arbitrary host read-write.</Person1>
<Person2>So it combined a known bug with a fix that never got a CVE?</Person2>
<Person1>Exactly. Then against upstream QEMU it found three 0-days and one patched bug missing from the distro. It ran about twelve hours, backtracking and writing minimal examples.</Person1>
<Person2>That is a tenant reading the landlord's public maintenance log, spotting a recalled lock and an unfiled repair note, and combining them to open the service door.</Person2>
<Person1>Right. Firecracker hardlocked the host but did not escape. The takeaway is an off-the-shelf VM is not a security boundary.</Person1>
<Person2>That VM escape is a containment risk. The money story is a different kind of risk. Nvidia projects seventy percent growth, about six hundred seventy three billion dollars in fiscal 2028 sales.</Person2>
<Person1>Wait, that would pass Apple and Alphabet by revenue?</Person1>
<Person2>Yes. Quarterly revenue hit ninety six point two billion, data center rose one seventeen percent to eighty nine billion. Supply, not demand, is the ceiling.</Person2>
<Person1>So memory shortages are capping it. And the new ACIE buyers are regional providers, startups, enterprises, not just hyperscalers?</Person1>
<Person2>Right. Think of it as a factory that also runs the bank. Nvidia funds OpenAI's Ohio campus and arranges up to five hundred billion in financing, which flows back as chip purchases.</Person2>
<Person1>That circular financing is the concern. If ACIE growth depends on Nvidia's balance sheet, the revenue is partly self-funded.</Person1>
<Person2>That circular financing is a demand story. On the other hand, a local inference teardown asks why your local LLM feels dumber than it is.</Person2>
<Person1>What did they actually isolate?</Person1>
<Person2>They pinned Qwen3.6-27B on one RTX PRO 6000 with BF16 and only swapped attention backends. FlashAttention 2, Flash Inference, and Triton start flipping top-1 tokens later in a 100k prompt.</Person2>
<Person1>Wait, same weights, same hardware?</Person1>
<Person2>Yes. Same backend runs were bit-identical, so the divergence is pure matrix multiply order. It is like three calculators that round at different steps. Same formula, but after enough operations the final digit flips.</Person2>
<Person1>And quantization made it worse?</Person1>
<Person2>KV cache int4 broke tool calling while BF16 and int8 recovered. The W8A16 quant beat first-party FP8. NVIDIA's NVFP4 hit about 50 percent flips by 88k and ran 'show run' instead of 'show arp'.</Person2>
<Person1>So the model isn't dumber. The stack is.</Person1>
<Person2>Right. Benchmarks need long-context tool calls, not zero-shot prompts. Use the model card sampler settings. And don't trust a low KLD claim without full methodology.</Person2>
<Person1>That stack-fidelity problem also appears in voice. Next up, Nari Labs got Qwen3-TTS to sub-50 ms p95 time-to-first-audio on one H100.</Person1>
<Person2>Wait, sub-50 ms while streaming? How?</Person2>
<Person1>They split Talker, Code Predictor, and Codec into three independently schedulable tasks under one scheduler. It prioritizes first audio, then playback deadlines.</Person1>
<Person2>So later chunks only need to arrive before the buffer empties.</Person2>
<Person1>Exactly. The scheduler anchors a batch with one urgent request and fills the rest with compatible work. They also capture the Code Predictor's fixed 15-step loop as one CUDA graph and cache Codec state.</Person1>
<Person2>It is like an emergency room that sends first-time patients straight to a doctor, then schedules follow-ups only when a prescription is about to run out.</Person2>
<Person1>Right. The payoff is ten requests per second at sub-50 ms p95, and about two dollars per million characters at full utilization. ElevenLabs V3 is one hundred dollars per million.</Person1>
<Person2>Wow. A fifty times cost gap, and better latency.</Person2>
<Person1>That fifty times cost gap is a good place to zoom out. Let's pull the whole episode together.</Person1>
<Person2>Right. The first theme is that capability is not just weights. The harness, the interface, and the context decide what surfaces.</Person2>
<Person1>Yeah. Prime Agent, PLVR, ASIL, Scroll, and the calibrated agent paper all showed the surrounding system decides what the model actually does. Meta^n and SRPO changed how recursion and credit flow.</Person1>
<Person2>Exactly. The second theme is execution. We saw Qwen reverse engineering locally, Gemini running lab hardware, and autonomous math producing new results.</Person2>
<Person1>That autonomy also raised the security and verification stakes. VM escape and prompt injection showed the boundary is not as solid as we thought.</Person1>
<Person2>Right. The third theme is economics and hardware. Redwood and AI with Authority design silicon, Jalapeño shifts efficiency, and Nvidia is buying distribution. The Reasoning Tax showed when that compute is worth it.</Person2>
<Person1>And the local LLM teardown plus the TTS work showed the deployment stack can make the same model feel smarter or dumber at very different cost.</Person1>
<Person2>So the arc goes from what a model knows, to what its system lets it do, to who can afford to run it.</Person2>
<Person1>That leaves a real question. If the harness, the hardware, and the economics all shape outcomes, what are we actually measuring when we say a model is capable?</Person1>
<Person2>That is the question to take into your own work this week.</Person2>
<Person1>Until next time, keep digging into the system around the model.</Person1>