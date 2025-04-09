import json
import re
import pickle
import numpy as np
import faiss
import sys # For exiting on critical errors
import time

# Core ML/AI Libraries
from sentence_transformers import SentenceTransformer, CrossEncoder
# Keep transformers import even if pipeline not used directly now
from transformers import pipeline
import google.generativeai as genai

# Visualization (Optional for Colab, might open new tab or render statically)
import plotly.graph_objs as go
from sklearn.manifold import TSNE
# import plotly.offline as pyo # Not typically needed in Colab for fig.show()
import os
from collections import defaultdict # Make sure this import is at the top of your script

# --- Configuration ---
# IMPORTANT: Replace with your actual API key securely
GEMINI_API_KEY = "AIzaSyCTdjPtjHrSYcU-_hLhSBAvnB0P9dKeDkc" # <<<--- ### REPLACE WITH YOUR KEY ### (Using placeholder for safety)
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or len(GEMINI_API_KEY) < 30: # Basic check
    print("🚨 ERROR: Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key in the code.")
    sys.exit(1) # Exit if key not set

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API Configured.")
except Exception as e:
    print(f"🚨 ERROR: Failed to configure Gemini API: {e}")
    sys.exit(1)

# --- Constants ---
EMBEDDING_MODEL_PKL = "rag_embedding_gpu.pkl"
GLOBAL_STORE_PKL = "global_store_gpu.pkl"
INSTRUCTION_DATA_JSON = "instruction_data.json"
# Load CrossEncoder FROM THE PICKLE FILE (as per previous steps)
CROSS_ENCODER_PKL = 'cross_encoder_gpu.pkl'
INITIAL_RETRIEVAL_K = 10 # Retrieve more candidates initially from FAISS
RE_RANKED_TOP_N = 5 # Use top N re-ranked results for LLM context
ISSUE_KEYWORDS = ["normal", "anxiety", "bipolar", "ptsd", "depression"]

# --- Color Mapping for Plot --- ADDED
issue_colors = {
    "normal": "#1f77b4",     # Strong blue
    "depression": "#ff7f0e", # Vivid orange
    "anxiety": "#9467bd",    # Medium purple
    "bipolar": "#8c564b",    # Muted brown
    "ptsd": "#17becf",       # Teal / Cyan
    "unknown": "#7f7f7f"     # Neutral gray
}

DEFAULT_PLOT_COLOR = "silver" # Fallback color if issue not in map

# --- Model Loading ---
# Load models ONCE at the start
def load_models():
    print("\n--- Loading Models ---")
    embedding_model = None
    cross_encoder = None
    gemini_model = None

    # Load Embedding Model
    try:
        with open(EMBEDDING_MODEL_PKL, "rb") as f:
            embedding_model = pickle.load(f)
        print(f"✅ Loaded embedding_model from {EMBEDDING_MODEL_PKL}")
    except FileNotFoundError:
        print(f"🚨 ERROR: Embedding model file not found: {EMBEDDING_MODEL_PKL}.")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 ERROR: Error loading embedding model: {e}")
        sys.exit(1)

    # Load CrossEncoder Model from Pickle
    try:
        with open(CROSS_ENCODER_PKL, "rb") as f:
            cross_encoder = pickle.load(f)
        print(f"✅ Loaded CrossEncoder model from pickle: {CROSS_ENCODER_PKL}")
        print("   ⚠️ Reminder: Loading models from pickle is less robust than standard methods.")
    except FileNotFoundError:
        print(f"🚨 ERROR: CrossEncoder pickle file not found: {CROSS_ENCODER_PKL}.")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 ERROR: Error loading CrossEncoder model from pickle: {e}")
        sys.exit(1)

    # Configure Gemini Model
    try:
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        print("✅ Configured Gemini model (gemini-1.5-flash)")
    except Exception as e:
        print(f"🚨 ERROR: Failed to configure Gemini model: {e}")
        sys.exit(1)

    print("--- Models Loaded Successfully ---")
    return embedding_model, cross_encoder, gemini_model

embedding_model, cross_encoder, gemini_model = load_models()


# --- Data Loading ---
# Load data store content (can be reloaded if updated)
def load_global_store():
    print(f"\n--- Loading Knowledge Base from {GLOBAL_STORE_PKL} ---")
    try:
        with open(GLOBAL_STORE_PKL, "rb") as f:
            store = pickle.load(f)
        required_keys = {"documents", "outputs", "index", "embeddings"}
        if not required_keys.issubset(store.keys()):
            print(f"🚨 ERROR: Global store file {GLOBAL_STORE_PKL} is missing required keys.")
            return None, None, None, None, None # Indicate failure + metadatas
        documents = store.get("documents", [])
        outputs = store.get("outputs", [])
        index = store.get("index")
        embeddings = store.get("embeddings")
        print(f"🔁 Loaded {len(documents)} documents.")
        if not isinstance(index, faiss.Index):
             print("   ⚠️ Warning: Loaded 'index' might not be a valid Faiss index.")
        elif index.ntotal == 0:
             print("   ⚠️ Warning: Faiss index is empty.")

        # --- Extract Metadata --- ADDED HERE
        metadatas = []
        print("   -> Extracting metadata (issue) by searching keywords in first sentence...")

        for document in documents:
            issue = "unknown" # Default issue

            if document and isinstance(document, str): # Check if document is a non-empty string
                # Extract first sentence (or whole doc if no period) and lowercase it
                first_period_index = document.find('.')
                if first_period_index != -1:
                    # Extract up to and including the first period
                    first_sentence = document[:first_period_index + 1].lower()
                else:
                    # If no period, search the whole document (lowercased)
                    # Alternatively, you could take the first N characters: document[:100].lower()
                    first_sentence = document.lower()

                # Search for keywords in the defined order
                for keyword in ISSUE_KEYWORDS:
                    # Use simple 'in' check (case-insensitive due to lowercasing first_sentence)
                    if keyword in first_sentence:
                        issue = keyword # Assign the first keyword found
                        break # Stop searching once a keyword is found

            metadatas.append({"issue": issue})

        # --- End Metadata Extraction ---

        return documents, outputs, index, embeddings, metadatas # Return metadatas too

    except FileNotFoundError:
        print(f"🚨 WARNING: Global store file not found: {GLOBAL_STORE_PKL}. KB is empty.")
        return [], [], None, None, [] # Return empty lists/None
    except Exception as e:
        print(f"🚨 ERROR: Error loading global store: {e}")
        return None, None, None, None, None # Indicate failure


# --- Functions ---

# Append new input and response (Updates files directly)
def append_to_json_file(instruction, input_text, output, filename=INSTRUCTION_DATA_JSON, store_filename=GLOBAL_STORE_PKL):
    # (Content of this function remains the same as previous version)
    print("\n--- Updating Knowledge Base ---")
    try:
        print(f"📄 Reading/Appending {filename}...")
        try:
            with open(filename, "r") as file: content = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"   -> {filename} not found or invalid. Starting fresh."); content = []
        new_record = {"instruction": instruction, "input": input_text, "output": output}
        content.append(new_record)
        with open(filename, "w") as file: json.dump(content, file, indent=4)
        print(f"   -> Record appended to {filename}")

        print("📚 Rebuilding internal lists...")
        current_documents = [f"{item.get('instruction','')} {item.get('input','')}" for item in content]
        current_outputs = [item.get("output", "") for item in content]
        print(f"   -> Total Records: {len(current_documents)}")

        if not current_documents:
            print("   -> No documents to index."); current_index = None; current_embeddings = None
        else:
            print("🔍 Re-encoding documents...");
            try:
                new_embeddings = np.array([embedding_model.encode(doc) for doc in current_documents]).astype("float32")
            except Exception as e: print(f"   -> 🚨 ERROR: Failed to encode documents: {e}"); return False
            print("📦 Building new FAISS Index...")
            if new_embeddings.shape[0] > 0:
                dimension = new_embeddings.shape[1]; current_index = faiss.IndexFlatL2(dimension)
                current_index.add(new_embeddings); current_embeddings = new_embeddings
                print(f"   -> FAISS Index rebuilt ({current_index.ntotal} vectors, Dim: {dimension})")
            else: print("   -> No embeddings generated, index not rebuilt."); current_index = None; current_embeddings = None

        print(f"💾 Saving updated global store to {store_filename}...")
        updated_store = {"documents": current_documents, "outputs": current_outputs, "index": current_index, "embeddings": current_embeddings}
        with open(store_filename, "wb") as f: pickle.dump(updated_store, f)

        print("--- Knowledge Base Update Complete! ---"); return True
    except Exception as e: print(f"🚨 ERROR: An error occurred during KB update: {e}"); return False


# --- Hover Text Formatting Functions ---

# Original function for TERMINAL output (uses newline \n)
def format_terminal_hover_text(text, idx, max_len=200, line_width=70):
    text = str(text) if text is not None else ""
    truncated = text[:max_len] + ('...' if len(text) > max_len else '')
    wrapped_lines = [truncated[i:i+line_width] for i in range(0, len(truncated), line_width)]
    wrapped = '\n'.join(wrapped_lines)
    return f"Index: {idx}\nText:\n{wrapped}"

# NEW function for PLOTLY hover text (uses <br>)
def format_plot_hover_text(text, idx, issue="N/A", max_len=200, line_width=50):
    text = str(text) if text is not None else ""
    truncated = text[:max_len] + ('...' if len(text) > max_len else '')
    # Replace potential HTML tags in text to avoid breaking hover layout
    safe_truncated = truncated.replace('<', '&lt;').replace('>', '&gt;')
    wrapped_lines = [safe_truncated[i:i+line_width] for i in range(0, len(safe_truncated), line_width)]
    wrapped = '<br>'.join(wrapped_lines)
    return f"<b>Index: {idx}</b><br>Issue: {issue}<br>Text:<br>{wrapped}"

# NEW function for PLOTLY query hover text (uses <br>)
def format_plot_hover_query(text, label="Query", max_len=200, line_width=50):
    text = str(text) if text is not None else ""
    truncated = text[:max_len] + ('...' if len(text) > max_len else '')
    safe_truncated = truncated.replace('<', '&lt;').replace('>', '&gt;')
    wrapped_lines = [safe_truncated[i:i+line_width] for i in range(0, len(safe_truncated), line_width)]
    wrapped = '<br>'.join(wrapped_lines)
    return f"<b>{label}:</b><br>{wrapped}"
# --- End Hover Text Functions ---

# Ryff filtering (Ensure input is string) - No changes needed functionally
def filter_output_by_ryff(output_text, selected_ryff):
    # (Content of this function remains the same as previous version)
    if not selected_ryff: return str(output_text) if output_text is not None else ""
    output_text = str(output_text) if output_text is not None else ""
    filtered = []
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', output_text) if s]
    for sentence in sentences:
        for param in selected_ryff:
            if param.lower() in sentence.lower(): filtered.append(sentence.strip()); break
    return " ".join(filtered) if filtered else "⚠️ No insights found for selected Ryff parameters."

# --- Main Interaction Loop ---
def main():
    while True: # Loop for multiple queries
        # Load the latest data AND METADATA at the start of each iteration
        documents, outputs, index, embeddings, metadatas = load_global_store()

        # Critical check after loading
        if documents is None:
             print("🚨 Exiting due to failure loading knowledge base.")
             break
        if not documents:
             print("⚠️ Knowledge base is currently empty.")
             # Optionally allow adding data here or just prompt for query anyway

        print("\n============================================")
        print("🧠 Advanced Wellbeing Insight Generator")
        print("============================================")

        # --- Get User Input ---
        print("📝 Describe your situation or thoughts:")
        text_input = input("> ")
        if not text_input: print("⚠️ Input cannot be empty. Please try again."); continue
        print("\n💭 Mental health issue (e.g., anxiety, depression):")
        mental_issue = input("> ")
        if not mental_issue: print("⚠️ Mental issue cannot be empty. Please try again."); continue

        # Ryff Parameter Selection (Logic remains the same)
        ryff_params = ["Autonomy", "Environmental Mastery", "Personal Growth", "Positive Relations", "Purpose in Life", "Self-Acceptance"]
        selected_ryff = []
        print("\n🎯 Optional: Select up to 3 Ryff parameters to focus insight."); print("   Available parameters:")
        for i, param in enumerate(ryff_params): print(f"     {i+1}. {param}")
        print("   Enter numbers separated by commas (e.g., 1,3,5) or press Enter for all:"); ryff_selection = input("> ").strip()
        if ryff_selection:
            try:
                selected_indices = [int(x.strip()) - 1 for x in ryff_selection.split(',') if x.strip()]
                if len(selected_indices) > 3: print("   -> Warning: Too many selections. Using the first 3."); selected_indices = selected_indices[:3]
                selected_ryff = [ryff_params[i] for i in selected_indices if 0 <= i < len(ryff_params)]
                if selected_ryff: print(f"   -> Selected Ryff parameters: {', '.join(selected_ryff)}")
                else: print("   -> No valid parameters selected. Will use all."); selected_ryff = ryff_params # Default to all if input invalid
            except (ValueError, IndexError): print("   -> Invalid input. Will use all Ryff parameters."); selected_ryff = ryff_params
        else: print("   -> No parameters selected. Will provide insight for all Ryff parameters."); selected_ryff = ryff_params
        print("===========================================")


        # --- Input Validation for KB Index ---
        if index is None or not hasattr(index, 'ntotal') or index.ntotal == 0:
             print("\n🚨 ERROR: Knowledge Base Index is empty or invalid. Cannot perform retrieval.")
             try_again = input("Knowledge base empty. Try another query? (y/n): ").lower()
             if try_again != 'y': break
             else: continue

        # --- RAG Process ---
        # (Steps 1: Query, 2: Retrieval, 3: Re-ranking remain the same)
        print("\n⏳ Processing: Retrieving -> Re-ranking -> Generating...")
        start_rag_time = time.time()
        query = f"Provide wellbeing insight for the below text with {mental_issue}. {text_input}" # Step 1

        # Step 2: Initial Retrieval
        print("   Step 1: Initial Retrieval from Knowledge Base...")
        try:
            query_embedding = np.array([embedding_model.encode(query)]).astype("float32"); k_initial = min(INITIAL_RETRIEVAL_K, index.ntotal)
            distances, initial_indices = index.search(query_embedding, k_initial); initial_indices = initial_indices[0]; distances = distances[0]
            if len(initial_indices) == 0: print("   ⚠️ No documents found in initial retrieval."); continue
            print(f"   -> Retrieved {len(initial_indices)} candidates.")
        except Exception as e: print(f"   -> 🚨 ERROR during initial retrieval: {e}"); continue

        # Step 3: Re-ranking
        print("   Step 2: Re-ranking candidates for relevance...")
        try:
            retrieved_docs_text = [documents[i] for i in initial_indices]; cross_inp = [[query, doc_text] for doc_text in retrieved_docs_text]
            cross_scores = cross_encoder.predict(cross_inp); reranked_results = sorted(zip(initial_indices, cross_scores), key=lambda x: x[1], reverse=True)
            n_final = min(RE_RANKED_TOP_N, len(reranked_results)); reranked_indices = [idx for idx, score in reranked_results[:n_final]]; reranked_scores = [score for idx, score in reranked_results[:n_final]]
            if not reranked_indices: print("   -> ⚠️ Re-ranking resulted in zero candidates."); continue
            print(f"   -> Re-ranked and selected top {len(reranked_indices)} candidates.")
        except Exception as e:
            print(f"   -> 🚨 ERROR during re-ranking: {e}"); print("      Falling back to top results from initial retrieval.")
            n_final = min(RE_RANKED_TOP_N, len(initial_indices)); reranked_indices = initial_indices[:n_final]; reranked_scores = [1.0 - d for d in distances[:n_final]]

        # Step 4: Prepare Context & Display Matches (Using TERMINAL hover function here)
        print("   Step 3: Preparing context for Generation...")
        combined_context_for_llm = ""; print("\n--- Top Re-ranked Records Used as Context ---")
        context_parts = []
        for i, idx in enumerate(reranked_indices):
             try:
                 original_input_text = documents[idx]; original_output_text = outputs[idx]
                 filtered_output = filter_output_by_ryff(original_output_text, selected_ryff)
                 context_parts.append(f"--- Context Source {i+1} (Score: {reranked_scores[i]:.4f}) ---\n{filtered_output}")
                 # Use TERMINAL format for printing details
                 print(f"\n### Match {i+1} (Index: {idx}, Score: {reranked_scores[i]:.4f}) ###")
                 print(f"  Retrieved Input (Instruction+Situation):\n    {format_terminal_hover_text(original_input_text, idx)}") # Use terminal format
                 print(f"  Retrieved Output (Filtered for Context):\n    {filtered_output}")
             except IndexError: print(f"   -> Warning: Index {idx} out of bounds. Skipping."); continue
             except Exception as e: print(f"   -> Warning: Error processing context index {idx}: {e}. Skipping."); continue
        combined_context_for_llm = "\n\n".join(context_parts); print("---------------------------------------------")

        # Step 5: Generation (Prompt slightly adjusted for clarity)
        if selected_ryff == [] :
            selected_ryff = ryff_params
        print("\n   Step 4: Generating final insight with LLM...")
        ryff_prompt_part = f"exactly focusing on the Ryff parameters: {', '.join(selected_ryff)}" if selected_ryff != ryff_params else "considering general psychological wellbeing principles" # Adjust if all were implicitly selected
        num_lines_requested = max(len(selected_ryff) if selected_ryff != ryff_params else 1, 1) # Adjust if all selected

        prompt = f"""
        You are part of a Retrieval-Augmented Generation (RAG) system designed to provide psychological wellbeing insights.

        Given:
        - User's Input Text: "{text_input}"
        - User's Mental Health Issue: "{mental_issue}"
        - Relevant Context (Ranked by Relevance):
        --- BEGIN CONTEXT ---
        {combined_context_for_llm}
        --- END CONTEXT ---

        Task:
        Using the provided context, generate exactly {num_lines_requested} distinct and insightful bullet points — one **for each** of the following Ryff Psychological Wellbeing parameters: {selected_ryff}.

        Instructions:
        - Each bullet point must correspond to a **different** parameter from the list above (maintain the same order).
        - Be empathetic, supportive, and constructive in tone.
        - Do **not** repeat the context or user's input verbatim.
        - Address the user's issue in a personalized and context-aware manner.
        - Do not add extra bullets or summaries. Just output exactly {num_lines_requested} bullet points.

        Begin:
        """

        try:
            response = gemini_model.generate_content(prompt)
            # (Response parsing logic remains the same)
            if hasattr(response, 'text') and response.text: generated_output = response.text.strip()
            elif hasattr(response, 'parts') and response.parts: generated_output = "".join(part.text for part in response.parts).strip()
            else: generated_output = str(response).strip();
            if not generated_output or len(generated_output) < 10: print("   -> ⚠️ Warning: Received potentially empty/invalid LLM response."); generated_output = "⚠️ Received no valid text content from LLM."
            final_output = generated_output
        except Exception as e: print(f"   -> 🚨 ERROR during Gemini generation: {e}"); final_output = "⚠️ Could not generate insight due to an API error."
        end_rag_time = time.time(); print(f"   -> Insight generation took {end_rag_time - start_rag_time:.2f} seconds.")

        # Step 6: Display Final Output (Remains the same)
        print("\n============================================"); print("✅ Results:"); print("============================================")
        print("\n📘 Summary of your situation:"); print(f"   {text_input}")
        print("\n🧠 Advanced Wellbeing Insight:"); print(f"   {final_output}")
        print("\n🔗 Top Re-ranked Match Score:"); print(f"   {reranked_scores[0]:.4f} (CrossEncoder score, higher is better)"); print("--------------------------------------------")

        # --- Visualization Section (MODIFIED) ---
        plot_choice = input("\n📊 Generate 3D visualization? (y/n - might open new tab/be slow): ").lower()
        if plot_choice == 'y':
            print("   Step 5: Generating Visualization...")
            try:
                print("      -> Calculating t-SNE (this can take a moment)...")
                if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] == 0:
                    print("      -> ⚠️ KB embeddings missing/invalid. Cannot plot.")
                elif len(metadatas) != embeddings.shape[0]: # Check metadata length matches embeddings
                     print(f"      -> ⚠️ Metadata length ({len(metadatas)}) doesn't match embeddings count ({embeddings.shape[0]}). Cannot color plot correctly.")
                else:
                    query_embedding_2d = query_embedding.reshape(1, -1)
                    combined_embeddings = np.vstack([query_embedding_2d, embeddings])
                    n_samples = combined_embeddings.shape[0]
                    perplexity_val = min(30, max(5, n_samples - 1)) # Ensure perplexity is valid range

                    if perplexity_val <= 4: print(f"      -> ⚠️ Not enough data points ({n_samples}) for meaningful t-SNE plot.")
                    else:
                        tsne_iter = max(250, min(1000, int(n_samples * 2.5))) # Faster iterations
                        reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity_val, max_iter=tsne_iter, init='pca', learning_rate='auto', n_jobs=-1)
                        reduced_all = reducer.fit_transform(combined_embeddings)

                        query_3d = reduced_all[0]
                        all_3d = reduced_all[1:] # KB points
                        valid_reranked_indices = [i for i in reranked_indices if i < len(all_3d)]
                        if len(valid_reranked_indices) != len(reranked_indices): print("      -> Warning: Some re-ranked indices out of bounds.")
                        topk_3d = np.array([all_3d[i] for i in valid_reranked_indices]) if valid_reranked_indices else np.empty((0,3))

                        plot_traces = [] # Initialize list to hold all plot traces
                        issues_in_plot = set() # Keep track of issues plotted

                        # 1. Group original indices by issue
                        print("      -> Grouping data points by issue for legend...")
                        grouped_points = defaultdict(list)
                        # Ensure metadatas length matches all_3d length for accurate grouping
                        if len(metadatas) == len(all_3d):
                            for i in range(len(all_3d)):
                                # 'i' here is the index within all_3d, which corresponds to the original KB index
                                issue = metadatas[i].get("issue", "unknown") # Get issue from corresponding metadata
                                grouped_points[issue].append(i) # Store original KB index 'i'
                        else:
                            print(f"      -> ⚠️ ERROR: Mismatch between metadata ({len(metadatas)}) and embedding points ({len(all_3d)}). Cannot create accurate legend.")
                            # As a fallback, you might add a single generic KB trace here if needed,
                            # otherwise the plot might lack KB points if this error occurs.
                            # Example Fallback (Optional):
                            plot_traces.append(go.Scatter3d(x=all_3d[:,0], y=all_3d[:,1], z=all_3d[:,2], mode='markers', name='KB (Error)', marker=dict(size=3, color='pink')))

                        # 2. Create a separate trace for each issue type found in KB
                        print("      -> Preparing traces per issue...")
                        for issue, original_indices in grouped_points.items():
                            if not original_indices: continue # Skip if somehow an issue has no points

                            # Select the 3D points and other data corresponding to these original indices
                            # Note: 'original_indices' contains indices relative to the original KB/embeddings/metadata list
                            # 'all_3d' is the t-SNE result EXCLUDING the query point, so its indices match original_indices
                            issue_points_3d = all_3d[original_indices]
                            issue_color = issue_colors.get(issue, DEFAULT_PLOT_COLOR) # Get color for this issue

                            # Create hovertext specifically for these points using the original indices
                            issue_hovertext = [format_plot_hover_text(documents[i], i, issue) for i in original_indices]

                            issues_in_plot.add(issue)
                            # Create the trace for this specific issue
                            issue_trace = go.Scatter3d(
                                x=issue_points_3d[:, 0], y=issue_points_3d[:, 1], z=issue_points_3d[:, 2],
                                mode='markers',
                                name=issue.capitalize(), # Legend name = Issue name (e.g., "Anxiety")
                                marker=dict(
                                    size=3,
                                    color=issue_color, # Assign the single specific color
                                    opacity=0.5 # Slightly more opaque for visibility
                                ),
                                hoverinfo='text',
                                hovertext=issue_hovertext # Assign specific hover text
                            )
                            plot_traces.append(issue_trace) # Add this issue's trace to the list

                        print(f"      -> Created KB traces for issues: {', '.join(sorted(list(issues_in_plot)))}")

                        # Re-ranked Top N points - USE PLOT HOVER TEXT
                        topk_trace = go.Scatter3d(
                            x=topk_3d[:, 0], y=topk_3d[:, 1], z=topk_3d[:, 2], mode='markers+text', name=f'Top {len(valid_reranked_indices)}',
                            marker=dict(size=7, color='green', symbol='circle'), text=[f"R{i+1}" for i in range(len(valid_reranked_indices))], textposition='top center',
                            hoverinfo='text',
                            hovertext=[format_plot_hover_text(documents[idx], idx, metadatas[idx].get("issue", "unknown")) for idx in valid_reranked_indices] # Use PLOT hover func
                        )
                        # Query point - USE PLOT HOVER TEXT
                        query_trace = go.Scatter3d(
                            x=[query_3d[0]], y=[query_3d[1]], z=[query_3d[2]], mode='markers+text', name='Your Query',
                            marker=dict(size=11, color='red'), text=["Q"], textposition="middle center",
                            hoverinfo='text',
                            hovertext=[format_plot_hover_query(query, "Query")] # Use PLOT query hover func
                        )
                        # Lines (remain the same)
                        lines = [go.Scatter3d(x=[query_3d[0], p[0]], y=[query_3d[1], p[1]], z=[query_3d[2], p[2]], mode='lines', line=dict(color='darkgreen', width=2, dash='dot'), showlegend=False, hoverinfo='none') for p in topk_3d]

                        # Combine traces and layout
                        # fig = go.Figure(data=[kb_trace, topk_trace, query_trace] + lines)
                        fig = go.Figure(data=plot_traces + [topk_trace, query_trace] + lines)
                        fig.update_layout(
                             title=f"3D t-SNE: Query → Re-ranked Top {len(valid_reranked_indices)} (Colored by Issue)",
                             scene=dict(xaxis_title='TSNE-1', yaxis_title='TSNE-2', zaxis_title='TSNE-3'),
                             legend=dict(x=0, y=1, traceorder='reversed', title='Legend'), # Added legend title
                             margin=dict(l=0, r=0, b=0, t=40),
                             # height=750,
                             hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1)
                         )

                        print("      -> Showing plot... (Check for a new tab or output below)")
                        fig.show() # Display the plot

            except ImportError: print("      -> ⚠️ Plotly or Scikit-learn not installed. Skipping visualization.")
            except Exception as e: print(f"      -> 🚨 ERROR generating visualization: {e}")
        # --- End Visualization ---


        # Step 8: KB Update (Logic remains the same)
        print("\n--- Knowledge Base Update ---")
        if "Could not generate insight" not in final_output and "error" not in final_output.lower():
             update_choice = input("💾 Save this generated insight to the knowledge base? (y/n): ").lower()
             if update_choice == 'y':
                ryff_focus = f"focusing on {', '.join(selected_ryff)}" if selected_ryff != ryff_params else "with general focus"
                to_append_instruction = f"Provide wellbeing insight for situation related to '{mental_issue}' {ryff_focus}."
                success = append_to_json_file(to_append_instruction, text_input, final_output)
                if success: print("   -> Knowledge base update requested and processed.")
                else: print("   -> Knowledge base update failed.")
             else: print("   -> Skipping knowledge base update.")
        else: print("   -> Skipping knowledge base update due to previous generation error.")


        # --- Loop Control ---
        print("\n--------------------------------------------")
        another_query = input("Perform another query? (y/n): ").lower()
        if another_query != 'y': print("\nExiting application. Goodbye!"); break

if __name__ == "__main__":
    main()
