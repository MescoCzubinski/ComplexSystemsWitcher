# ComplexSystemsWitcher

Graph analysis of character co-occurrences in the Witcher book series.

## Requirements

```
networkx
matplotlib
numpy
```

## Usage

```
python connections.py   # build data/connections.csv from books/
python graph.py         # draw the graph → data/graph.png
python statistics.py    # print centrality analysis (order/size, centrality, matrices)
```

## Files

| File | Description |
|------|-------------|
| `graph.py` | Draws the co-occurrence graph |
| `statistics.py` | Centrality analysis (degree, closeness, betweenness) |
| `connections.py` | Builds `connections.csv` from raw book text |
| `data/connections.csv` | Edge list with co-occurrence weights |
| `data/graph.png` | Output visualization |

## Graph properties

- **Nodes**: Witcher characters
- **Edges**: co-occurrence within the same sentence
- **Weight**: number of sentences two characters share
- Only edges with `weight >= 5` are shown in the visualization
