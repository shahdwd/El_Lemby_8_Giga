#   
**In short:** Qanony went from following a fixed set of steps to researching each question itself, and it now checks that its answers match the law it cites.  
  
## What we achieved  
  
| Feature | What it does now | Why it matters |  
|---|---|---|  
| **Research agent** | Chooses its own tools to search the law, open an exact article or explore related articles | This is what makes the project agentic, which was the workshop's goal |  
| **Answer verifier** | Reads the cited articles and checks they say what the answer claims | Stops real articles being misused, like Art. 152 given as the penalty for theft |  
| **Honest confidence** | An answer with no verified citations is always low confidence | It no longer looks sure when it has nothing to back it up |  
| **Research steps in the UI** | Shows which tools the agent used and what they found | Users can see how an answer was reached, which also helps in a demo |  
| **Harder eval** | Now has 17 cases, adding fraud, assault and murder, and saves every answer | We can check how it handles laws we don't have |  
| **Safe fallback** | If the agent fails, the old fixed flow answers instead | One agent failure doesn't take the app down |  
  
**Results:** it scored 11/11 on legal questions and 6/6 on safety cases in all three eval runs.  
  
**The cost:** answers now take about 11 seconds on average, up from 5.9.  
  
## What's next  
I'd do these in order:  
  
1. **Stabilize the contract demo.** This is the biggest risk for a live demo: in 3 test runs, one took 62 seconds and one missed the dismissal clause. The answers also apply Art. 156 (open-ended contracts) to a fixed-term contract.  
2. **Test it in the browser.** The verifier has only been tested through scripts and the eval, so run the full demo in the UI, including a real Arabic PDF upload.  
3. **Fix the fallback message.** It should say plainly when a law isn't in our data, and stop listing unrelated articles.  
4. **Polish the research steps.** Show them live while the agent works, and translate them for the Arabic UI.  
5. **Add the two agent tools.** `ask_user` would let it ask for missing facts, and `calculate_entitlement` would work out things like leave or severance. Together they would make it look much more agentic, because right now it usually does just one search on the eval questions.  
6. **Get the missing laws.** Theft, fraud, assault and murder can't be answered without the full Penal Code, and Criminal Procedure and the Commercial Code are also mostly missing. This would help real users most, but it depends on finding good source files.  
7. **Prepare the pitch.** Build it around the theft example and save 2–3 good answers in case the internet fails.  
  
If the demo is soon, focus on 1, 2 and 7. If you have more time, 5 adds the most to the agentic story.  
