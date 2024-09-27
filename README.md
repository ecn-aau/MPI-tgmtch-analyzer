## Optimistic Tag Matching message rate plot generation
- Navigate to `exampi/plots` folder
- Run `generation.ipynb` notebook to generate Figure 8 on the text

## MPI analyzer + plots
- Navigate to `scripts` folder
- Run `run.sh <traces>` where `<traces>` is the folder with all the traces.
    - If the project structure is maintained, the command should be `./run.sh ../traces`
    - To obtain the same traces used in the paper, email <jsg@es.aau.dk>, as they are aroun 7 GiB in size, they cannot be uploaded into GitHub.
- After all the traces have been processed, run `plots.ipynb` notebook to generate, among others, Figures 6 and 7 on the text
