# ARES: Automatic Robot Evaluation System

<img src="assets/ares_system_diagram.png" alt="ARES System Diagram"/> 

ARES (Automatic Robot Evaluation System) is a open-source (Apache 2.0) platform for automatically ingesting, curating, and evaluating robot data using ML models to quickly and accurately understand policy performance, identify areas for improvement, and generate new robot datasets, all without setting up any heavy infrastructure.
Researchers tend to chase point-solutions for specific tasks or paper implmentations, but ARES is designed to be a generalized platform for long-term robot research that scales from a laptop to the cloud.
ARES is built to be simple and scalable, with a special focus on ease of use. All computation and model inference can be run through local resources or cloud APIs via model providers like OpenAI, Anthropic, Gemini, Modal, Replicate, etc., requiring only a credit card for access - no complex cloud infrastructure or GPU setup needed. 

At a high level, ARES is composed of three main components: 
- Ingestion: automatically transform raw robot data into a structured format with VLMs
- Annotation: annotate the rollouts with pseduo-labels for downstream tasks
- Curation and Modeling: understand data distributions and select data for training or evaluation

[Demo Video](https://example.com/demo) TODO!

[Blog Post](https://example.com/blog) TODO! 

[Arxiv Paper](https://arxiv.org/) TODO!

## Who and what is ARES for?

ARES is a platform for understanding robot data, targeted at robot developers and researchers. Researchers and developers suffer from two main problems: building point-solutions for specific tasks or papers and transitioning from research-scripts to production-tools. ARES aims to solve both of these problems by providing a *platform* for robot data understanding, enabling rapid development of new robot systems.

You can use ARES to: 
- Curate and annotate ground-truth teleoperation data
- Evaluate the performance of robot models
- Analyze batches of robot data to improve policies

## Overview
- [🛠️ Stack](#stack)
- [🧠 Installation and Setup](#installation-and-setup)
- [👤 Configurations](#configurations) 
- [📊 Data](#data)
- [📥 Ingestion and Annotation](#ingestion-and-annotation)
- [🔍 Curation and Analysis](#curation-and-analysis)
- [🤖 Training and Export](#training-and-export)
- [📈 Evaluation](#evaluation)
- [🎉 Extras](#extras)
- [💰 Costs](#costs)
- [🔮 Limitations and Next Steps](#limitations-and-next-steps)
- [📝 Acknowledgements and Citation](#acknowledgements-and-citation)

## Stack
ARES is built to be a low-friction, easy-to-use infrastructure platform for robot research. As such, we select tools that are simple to setup locally but also smooth to scale to cloud-level resources.
- Databases: [MongoDB](https://www.mongodb.com/), [SQLAlchemy](https://www.sqlalchemy.org/), [FAISS](https://github.com/facebookresearch/faiss)
- Model Inference: [litellm](https://github.com/BerriAI/litellm), configured with your choice of model provider (OpenAI, Anthropic, Gemini, HuggingFace, etc.)
- Compute Orchestration: [Modal](https://modal.com/)
- Frontend: [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/)
- Development: [Docker](https://www.docker.com/), [Cursor](https://www.cursor.com/), [VSCode](https://code.visualstudio.com/), [Pydantic](https://docs.pydantic.dev/)

## Installation and Setup

### Prerequisites
First, clone the repository:
```bash
git clone https://github.com/jacobphillips99/ares.git
cd ares
```

If you want to use ARES as a package in your own project, follow the instructions below to install the requirements and install the package (add `-e` to install in editable mode and make changes). Otherwise, continue to the [Docker and Devcontainer Setup](#docker-and-devcontainer-setup) section to run ARES locally.
```bash
pip install -r requirements.txt
pip install -e .
```

### Docker and Devcontainer Setup
ARES was built with a Docker container in mind for easy setup, containerization, and deployment -- this makes it quick, easy, and consistent to setup ARES. We reccomend using a Docker container to run ARES, specifically utilizing a VSCode/Cursor devcontainer extension. The Dockerfile contains the necessary system packages for ARES, including a reference to the `requirements.txt` file for Python dependencies. Follow the steps below to setup Docker Desktop and the VSCode/Cursor devcontainer extension.

#### Docker Desktop Setup
Go to the [Docker Desktop](https://www.docker.com/products/docker-desktop/) website and download the latest version. Depending on your operating system, setup the Docker Desktop application. 

#### VSCode/Cursor Devcontainer Setup
Download and install the IDE of your choice. We recommend [VSCode](https://code.visualstudio.com/) or [Cursor](https://www.cursor.com/). Open the ARES directory in the IDE (using `File > Open Folder...`). Then use the Command Prompt to re-open the devcontainer using `Dev Containers: Reopen in Container` command. This should build the Docker container and start the ARES application, including opening an integrated terminal to the container. 

The `devcontainer.json` file contains the necessary configuration for the devcontainer, including the Dockerfile. It mounts some local directories into the container, such as the `data` directory for storing robot data, the `/tmp` directory for storing temporary files, and the `.cache/huggingface` directory for storing model weights.

### MongoDB Setup
In order to use the AnnotationDatabase, you will need to set up a MongoDB instance. We use `docker-compose` to run the MongoDB instance; see `mongo-docker-compose.yml` for the configuration. You can start the MongoDB instance by running
```bash
docker-compose -f mongo-docker-compose.yml up -d
```
in the root directory. This will start the MongoDB instance and expose it to the host machine on port 27017; the instance will automatically restart on container restart.

### Environment Variables
ARES uses environment variables to configure secrets like API keys. We mount these environment variables into the devcontainer using the `devcontainer.json` file. We copy over variables like API keys and credentials. If needed, you can also add your own certificates to the `/etc/ssl/certs` directory, which is also mounted into the container.

Once your IDE and environment are setup, you're ready to start using ARES!

## Configurations
All of ARES refers back to a base unit of data: the `Rollout` object defined in `ares/configs/base.py`. In reinforcement learning, a *rollout* is a collection of states, actions, and rewards in some environment. For our purposes, a `Rollout` is a collection of information about a robot completing a task, also called an episode.
These rollouts may be examples of a policy evaluation, a human teleoperation session, simulated or synthetic data, or other data formats. The `Rollout` object can contain multiple modalities of data, such as video, pointclouds, motor actions, and joint states. Rollouts can contain metadata about the episode, such as the dataset name, the robot configuration and embodiment, the data collection method, and more.
During ingestion, the user can provide hard-coded information about the rollout; other fields in the `Rollout` class will be provided by a VLM during ingestion. 

### Rollout
The `Rollout` class contains recursive subconfigurations: `Robot`, `Environment`, `Task`, and `Trajectory`. These subconfigurations may contain Pydantic fields or other subconfigurations. We use Pydantic not only to validate types, but also provide rich type information, such as descriptions, examples, regex patterns, and other numerical constraints. These `Field` objects are then used to automatically configure prompts for VLMs during ingestion, including the extra metadata from the Field. Fields with the suffix `_estimate` are inferred by the VLM during ingestion. The configuration classes can be recursively flattened or re-constructed into the original object; this enables a flexible system that can be used to create SQLModel types for database ingestion. Users can define their own configurations by inheriting from the base configuration class or adding `Field` objects to the configuration classes. 

<img src="assets/ares_configs.png" alt="Code example of ARES Rollout Configurations"/> 

## Data
We start with data from Tensorflow Datasets ports of the Open X-Embodiment project. While these datasets being open and available is great for general robot model training research, the iterator-style dataset makes it extremely difficult to do fast understanding and analysis, which motivates much of this work. As explained below in [Ingestion and Annotation](#ingestion-and-annotation), we can use Open X-Embodiment data or user-defined datasets. In order to download the Open X-Embodiment data, use the [`oxe_downloader` tool](https://github.com/mishmish66/oxe_downloader).

## Ingestion and Annotation
We adopt three main steps during ingestion: structured ingestion, embedding ingestion, and grounding ingestion. First, structured ingestion transforms the raw data into a structured format, which is then dumped into a SQL database. Second, embedding ingestion transforms data (such as the text description, video, action and state trajectories) into dense embeddings, which are then stored in a series of FAISS indexes. Third, grounding ingestion uses a VLM to detect objects in the scene and then detector and segmenter models to annotate the rollout with ground-truth labels and store these in a MongoDB database. 

We adopt the OpenXEmbodiment specification as the starting point for ingestion, allowing users to ingest datasets from the [Open X-Embodiment](https://robotics-transformer-x.github.io/) project. During ingestion, the user can provide hard-coded information about the episode, such as the natural language or templated task instructions. We load the raw data into an `OpenXEmbodimentEpisode` object, which includes Pydantic `model_validator` functions to process the raw data, such as swapping in `highres_image` or pulling `end_effector_cartesian_pos` from `state`. We also pull data from the [Open X-Embodiment spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0), which contains metadata about the datasets and is copied into the repository at `ares/extras/oxe.csv`.

The general ingestion pipeline can be found in `main.py`, which runs structured, embedding, and grounding ingestion. An example of ingesting a user-defined dataset can be found in `ares/scripts/pi_demo_ingestion.py`. This script contains the details on ingesting a series of demonstrations from a [Physical Intelligence blogpost](https://www.physicalintelligence.company/blog/pi0), hereafter referred to as the "PI Demos".

### Structured Ingestion
The script for structured ingestion can be found in `ares/scripts/run_structured_ingestion.py`. The script iterates through asynchronous batches of episodes, extracting structured, hard-coded information from each episode and "filling in the blanks" for estimated fields like `description_estimate`, `surface_estimate`, `success_estimate` etc. The information from the VLM populates a `Rollout` object, which is then flattened into the `RolloutSQLModel` object and dumped into the SQL database. Our `StrucutredDatabase` contains columns that match the recursively flattened `Rollout` object, enabling retrieval over the entire dataset. This yields long-term, structured metadata about the rollouts, which can be used for downstream analysis and curation.
```bash
VLM(video, task, Field Annotations) -> Rollout -> Flattened RolloutSQLModel -> SQL Database
```

### Embedding Ingestion
We are interested in finding *similar* rollouts across a dataset, amongst many axes: text description, task instruction, actions, states, etc. To enable this, we embed the rollouts into a dense vector space, where the euclidean distance between rollouts approximates their semantic similarity. See the script in `ares/scripts/run_trajectory_embedding_ingestion.py` for more details. For the text axes, we use a [Nomic Embedder](https://github.com/nomic-ai/nomic-embedder) to embed the text into a dense vector space. For the state and action axes, we first interpolate trajectories to a common time step, normalize sensor values per-dimension, and then flatten and embed the trajectories into a common vector space. This enables comparing and contrasting rollouts across different axes, such as finding rollouts in a similar `task` space but extremely different `action` spaces. 

### Grounding Ingestion
We want to make it as easy as possible to annotate rollouts with models. This can be further text descriptions (such annotating success criteria or grounding descriptions) or more traditional object detection and segmentation labels. During ingestion, we annotate at 5 FPS using `grounding-dino-tiny` and `sam-vit-base` models to detect objects and perform object segmentation. These annotations are stored in the AnnotationDatabase, which is backed by a MongoDB instance. The compute orchestration is handled by [Modal](https://modal.com/), which allows us to scale to cloud-level resources and perform asynchronous, parallelized inference. The script for grounding ingestion can be found in `ares/scripts/run_grounding.py`.

## Curation and Analysis
Once we've ingested and annotated a dataset, we can use ARES to curate and analyze the data. We provide a series of tools for visualizing and understanding the data in a simple frontend powered by Streamlit. You can run the frontend locally by running:

```bash
streamlit run ares/frontend/app.py
```

This opens a local Streamlit web app at `localhost:8501` that provides a high level overview of the ingested data, covering structured metadata, videos, annotations, and more. The ability to visualization annotations and retrieve rollouts by their annotations is a powerful tool for curation, as well as the ability to filter rollouts by their attributes. Retrieval over trajectory embeddings enables a powerful tool for finding in- and out-of-distribution rollouts. Exploring 2D projections of embeddings (such as task or description) enables the user to find clusters in the data, resulting in deeper understanding of the distributions.

### Structured and Unstructured Curation
The user can easily select a subset of rollouts based on their hard-coded or inferred attributes. For example, the user can select all rollouts with `dynamic background` and `low light` to understand the performance of the robot in low-light conditions with a dynamic background. Instead, the user can also select rollouts based on a UMAP projection of the unstructured attributes, such as the task instructions or inferred natural language description.

<img src="assets/ares_structured_curation.png" alt="Example of ARES structured curation"/>
<img src="assets/ares_unstructured_curation.png" alt="Example of ARES unstructured curation"/>

### Hero Plot
The main focus of the ARES platform is examining individual rollouts and finding similar examples. We provide a *hero display* to enable the user to view all information about a rollout at once, including video, annotations, other metadata. Additionally, the user can retrieve similar examples based on the rollout's embedding, searching for similar examples over task, description, or state and action trajectories.

<img src="assets/ares_hero_plot.png" alt="Example of ARES hero plot"/>
<img src="assets/ares_hero_plot_extras.png" alt="Example of ARES hero plot extra annotation and similarity retrieval"/>

### Trajectory Analysis
We can also examine trajectories to find in- and out-of-distribution rollouts. We view can find examples of difficult-to-reach joint angles or unusual action sequences. Researchers may use this tool to find poor performing regions of action or state spaces, or to find examples of successful trajectories that can be used to improve policies.

<img src="assets/ares_trajectories.png" alt="Example of ARES trajectory analysis"/>


## Training and Export
After curating and analyzing a dataset, we provide functionality for the user to export a curated slice. This export can be the graphical depiction of the dashboard saved to PDF or HTML; alternatively, you can export the data to CSV or Parquet files. These CSV and Parquet files can also be used as a pre-processsed dataset for training downstream models. We provide a `RolloutDataset` and `RolloutDatasetConfig` class to help train models using the ARES platform. See examples in `ares/training/preprocess.py` and `ares/training/train.py`.

## Evaluation
In order to decide which models should be used to power the ARES platform, we developed a benchmark based on the demonstrations released by [Physical Intelligence](https://www.physicalintelligence.company) in the [π₀ release](https://www.physicalintelligence.company/blog/pi0). The benchmark contains 20 videos over 11 tasks, most with both a success and failure rollout (as provided by the PI team). We use this benchmark to evaluate the performance of the ARES platform, as well as to select the best models for use inthe platform. We conduct evaluations over binary success classification, covering a range of VLMs, methods, consensus votes, and frames-per-second. We demonstrate the flexibility of the ARES platform by using the `VLM` object, creating a Pydantic `EvalConfig`, and launching batch asynchronous evaluations. In order to maintain consistent, robust evaluations, we first annotate `success_criteria` for each rollout using a VLM before predicting success. Evaluation results are summarized below, as plotted by `ares/notebooks/eval_nb.ipynb`.

We evaluate over: 
- **Models**: `claude-3.5-sonnet`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2-flash`, `gpt-4-turbo`, `gpt-4o`, and `gpt-4o-mini`
- **Methods**: `video` (feeding frames directly), `frame_descriptions` (describing each frame individually, then feeding all descriptions to the VLM)
- **Frames-per-second**: `0`, `0.25`, `0.5`, `1`, `2`, where `0` means just the first and final frames
- **Consensus**: Consensus of either the mean or median of the predictions
- **N Votes**: Number of votes to take for the consensus prediction, ranging over `1`, `2`, `3`, `4`, `5`

<img src="assets/ares_performance_vlm.png" alt="ARES Performance by VLM"/> 
<img src="assets/ares_performance_fps.png" alt="ARES Performance by FPS"/> 
<img src="assets/ares_performance_n.png" alt="ARES Performance by N Votes"/> 
<img src="assets/ares_performance_method.png" alt="ARES Performance by Method"/> 

We show `gpt-4o` to be the best performing model, with little impact from the number of votes or consensus strategy. Across models, performs seems to be mostly consistent after 0.5 FPS, while some older models actually perform better at very low FPS (e.g. `gemini-1.5-pro` performs best at `0` FPS). We find the `frame_description` method to actually slightly outperform the generalized `video` method, calling into question progress on long-context video benchmarks. However, this method is significantly more expensive and more difficult to run with respect to request-per-minute and token-per-minute rate limits. As such, we adopt the `video` method at 1 FPS with `gpt-4o` for all ARES data ingestion. 


## Extras
We provide some fun extra tools to help with robot research. 
- The `Rollout` and `RolloutSQLModel` classes can be used to dynamically flatten and reconstruct rollouts, enabled by hierarchical prefix matching. This enables smooth transitions between Pydantic and SQLModel types.
- The `ares/notebooks` directory contains a few notebooks for visualization; one for VLM results and the other for annotation results.
- The `ares/scripts` directory provides the top-level entrypoint for interacting with data in ARES, including the main ingestion pipeline. Other scripts include:
    - `ares/scripts/db_updaters/`: scripts for amending the `AnnotationDatabase` and `StrucutredDatabase`
    - `ares/scripts/annotating`: scripts for running annotations for in-context-learning, Embodied Chain of Thought, success criteria, and more, including the Modal-based grounding and detection scripts.
    - `ares/scripts/self_heal.py`: a tool for automatically syncing the `StructuredDatabase` with the `AnnotationDatabase` and `EmbeddingDatabase`. This means that there should never be ingested rollouts without annotations or embeddings.
- In order to provide robust error tracking during annotation, we provide a `ResultTracker` object that can be used to track the results of an annotation. This is useful for providing feedback on the status of an annotation job, as well as for providing a record of the annotation. See `ares/annotating/annotating_base.py` for more details.
- Normalization in ARES is dependent on the entire dataset being collected ahead of time. However, in practice, we often want to normalize data on-the-fly. To enable this, we provide the `NormalizationTracker` object to perform batch or online normalization. See `ares/databases/embedding_database.py` for more details.
 
## Costs
Ingesting, annotating, and annotating data can be quite expensive; we aim to use ARES to make robot research as accessible as possible. The original development of ARES was conducted solely on a laptop, using a combination of local resources (CPU and Disk) and cloud APIs. First, we mitigate costs by holding all data strucutres locally, including the `StrucutredDatabase`, `AnnotationDatabase`, and `EmbeddingDatabase`. Second, we downsample frames-per-second while ingesting, running `gpt-4o` at 1 FPS via API, using `grounding-dino-tiny` and `sam-vit-base` at 5 FPS via Modal, and the Nomic Embedder via local CPU. Back-of-the-envelope costs comes out to roughly 1 cent per rollout for `gpt-4o`, as most rollouts are relatively short. Annotating 5000 rollouts at 5 FPS (totaling >100,000 frames) for detection and segmentation costs less than $10 of $30 of monthly free Modal credits. We expect these costs to remain low as the pareto-frontier of efficicent multimodal models continues to bring down the cost of inference.


## Limitations and Next Steps
Right now, ARES is a platform to accelerate robot researchers. However, we need to acknowledge that current-generation VLMs are not perfect, leading to discrepencies or inaccuraies in the data during ingestion. At the same time, a lot of the questions being posed to the VLMs to understand robot data are relatively simple, so we are hopeful that next-generation VLMs will continue to improve in accuracy at long-context video understanding tasks. ARES is designed to be simple and scalable, so great next steps would be to make it easier to scale ARES into the cloud via hosted instances like Mongo Atlas, AWS, Snowflake, etc. On top of that, ingesting and open-sourcing more rollouts from the Open X-Embodiment project would be a great way to continue to expand the usefulness of the platform. Right now, we have ingested roughly 5000 rollouts from the roughly 1 million available, focusing on those datasets with natural language task annotations that can also fit on a laptop. Building stronger ingestion pipelines for more complex data types would be a great next step. Further next steps would refactor the ARES frontend (currently running on Streamlit) into a more performant system.


## Acknowledgements and Citation
This project was developed by [Jacob Phillips](jacobdphillips.com) as a part of the [Andreessen Horowitz American Dynamism Engineering Fellows program](https://a16z.com/the-american-dynamism-engineering-fellows-program/). Special thanks to the American Dynamism team for their support and feedback on the project.

If using the ARES platform in your work, please cite it to acknowledge the authors. Suggested format:

```bibtex
@software{ARES,
    title = {ARES: Automatic Robot Evaluation System},
    author = {Jacob Phillips},
    url = {https://github.com/jacobphillips99/ares/tree/main},
    version = {insert version number},
    date = {insert date of usage},
    year = {2025},
}
```

You can also cite the whitepaper or blog:
TODO TODO TODO


