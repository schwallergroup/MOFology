import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GPSConv, SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
import networkx as nx
import pandas as pd
import logging
import numpy as np
import gc
import rdflib
from rdflib import Graph
from tqdm import tqdm
from collections import defaultdict
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "GraphGPS")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mof_embeddings_gps.csv")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

class BigBirdAttention(nn.Module):
    """
    BigBird-style sparse attention for graphs.
    Uses block-sparse attention with:
    - Random attention (sparse)
    - Window attention (local neighbors)
    - Global attention (selected important nodes)
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, 
                 num_random_blocks=3, block_size=64, num_global_blocks=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        self.num_random_blocks = num_random_blocks
        self.block_size = block_size
        self.num_global_blocks = num_global_blocks
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query, key, value: (num_nodes, embed_dim)
            key_padding_mask: (num_nodes,) - True for valid nodes
        """
        num_nodes = query.size(0)
        batch_size = 1  # Single graph
        
        # Project to Q, K, V
        q = self.q_proj(query).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention: (num_heads, num_nodes, head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Compute attention scores with BigBird sparsity
        attn_output = self._bigbird_attention(q, k, v, num_nodes, key_padding_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(num_nodes, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout_layer(attn_output)
        
        return attn_output, None
    
    def _bigbird_attention(self, q, k, v, num_nodes, key_padding_mask):
        """
        Compute BigBird-style sparse attention.
        """
        num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # Initialize output
        attn_output = torch.zeros_like(v)
        
        # Block-based processing to reduce memory
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)
            block_size = end_idx - start_idx
            
            # Query block
            q_block = q[:, start_idx:end_idx, :]  # (num_heads, block_size, head_dim)
            
            # Attention scores for this block
            attn_scores = torch.zeros(
                num_heads, block_size, seq_len, device=device
            )
            
            # 1. Window attention (local neighbors) - simplified to nearby nodes
            window_size = min(self.block_size * 2, seq_len)
            window_start = max(0, start_idx - window_size // 2)
            window_end = min(seq_len, end_idx + window_size // 2)
            
            if window_end > window_start:
                k_window = k[:, window_start:window_end, :]
                qk_window = torch.matmul(q_block, k_window.transpose(-2, -1)) * self.scale
                attn_scores[:, :, window_start:window_end] = qk_window
            
            # 2. Random attention blocks
            if self.num_random_blocks > 0:
                num_random = min(self.num_random_blocks * self.block_size, seq_len)
                random_indices = torch.randperm(seq_len, device=device)[:num_random]
                k_random = k[:, random_indices, :]
                qk_random = torch.matmul(q_block, k_random.transpose(-2, -1)) * self.scale
                for i, idx in enumerate(random_indices):
                    attn_scores[:, :, idx] = qk_random[:, :, i]
            
            # 3. Global attention (first and last nodes, plus some random)
            global_indices = torch.tensor([0, seq_len - 1], device=device, dtype=torch.long)
            if self.num_global_blocks > 0:
                additional_global = torch.randperm(max(1, seq_len - 2), device=device)[:self.num_global_blocks] + 1
                global_indices = torch.cat([global_indices, additional_global])
            
            k_global = k[:, global_indices, :]
            qk_global = torch.matmul(q_block, k_global.transpose(-2, -1)) * self.scale
            for i, idx in enumerate(global_indices):
                attn_scores[:, :, idx] = qk_global[:, :, i]
            
            # Apply mask if provided
            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf')
                )
            
            # Softmax over sparse attention
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout_layer(attn_probs)
            
            # Compute weighted sum
            v_all = v  # (num_heads, seq_len, head_dim)
            attn_output_block = torch.matmul(attn_probs, v_all)
            attn_output[:, start_idx:end_idx, :] = attn_output_block
            
        return attn_output

class GPSModel(torch.nn.Module):
    def __init__(self, num_nodes, channels, pe_dim, num_layers=2, use_bigbird=True):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, channels)
        self.pe_lin = nn.Linear(pe_dim, channels)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = SAGEConv(channels, channels)
            
            if use_bigbird:
                # Use BigBird attention instead of default MultiheadAttention
                bigbird_attn = BigBirdAttention(
                    embed_dim=channels,
                    num_heads=4,
                    dropout=0.1,
                    num_random_blocks=3,
                    block_size=64,
                    num_global_blocks=2
                )
                conv = GPSConv(channels, local_conv, attn=bigbird_attn, dropout=0.1)
            else:
                conv = GPSConv(channels, local_conv, heads=4, dropout=0.1)
            
            self.convs.append(conv)
            
        self.post_lin = nn.Linear(channels, channels)

    def forward(self, x, pe, edge_index):
        x = self.node_emb(x) + self.pe_lin(pe)
        for conv in self.convs:
            x = conv(x, edge_index)
        return self.post_lin(x)

def train(model, data, optimizer, epochs=100, device='cpu'):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.pe, data.edge_index)
        pos_edge_index = data.edge_index
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=data.num_nodes)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean() - \
               torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def load_kg_from_ttl(kg_path):
    """Load the full knowledge graph from TTL file and convert to NetworkX."""
    logging.info(f"Loading Knowledge Graph from {kg_path}...")
    g = Graph()
    g.parse(kg_path, format="turtle")
    logging.info(f"Loaded {len(g)} triples.")
    
    # Build NetworkX graph from RDF triples
    logging.info("Building NetworkX graph from RDF triples...")
    nx_graph = nx.Graph()
    for s, p, o in tqdm(g, desc="Converting to NetworkX"):
        if isinstance(o, (rdflib.URIRef, rdflib.BNode)):
            nx_graph.add_edge(str(s), str(o))
    
    del g
    gc.collect()
    
    logging.info(f"NetworkX graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")
    return nx_graph

def extract_mof_metadata(kg_path):
    """Extract MOF metadata from the TTL file."""
    logging.info("Extracting MOF metadata from KG...")
    g = Graph()
    g.parse(kg_path, format="turtle")
    
    # Query for MOFs and their basic metadata
    query = """
    PREFIX : <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?csd ?prop ?linker_smiles ?metal
    WHERE {
        ?mof a ?type .
        FILTER(?type IN (:MOF, :ExperimentalMOF, :HypotheticalMOF))
        OPTIONAL { ?mof :hasCSDCode ?csd }
        OPTIONAL { 
            ?mof :hasStructuralProperty ?prop .
            FILTER(CONTAINS(STR(?prop), "SpaceGroup")) 
        }
        OPTIONAL { 
            ?mof :hasLinker ?linkerObj .
            ?linkerObj :hasSMILES ?linker_smiles .
        }
        OPTIONAL { 
            ?mof :hasMetalNode ?metalObj .
            BIND(STR(?metalObj) AS ?metal)
        }
    }
    """
    results = list(g.query(query))
    mof_dict = {}
    
    for row in tqdm(results, desc="Processing MOF metadata"):
        mof_uri = str(row.mof)
        if mof_uri not in mof_dict:
            mof_dict[mof_uri] = {
                "mof_uri": mof_uri,
                "csd_code": str(row.csd) if row.csd else None,
                "space_groups": set(),
                "linkers": set(),
                "metals": set()
            }
        if row.prop:
            sg = str(row.prop).split("#")[-1]
            mof_dict[mof_uri]["space_groups"].add(sg)
        if row.linker_smiles:
            mof_dict[mof_uri]["linkers"].add(str(row.linker_smiles))
        if row.metal:
            metal = str(row.metal).split("#")[-1]
            mof_dict[mof_uri]["metals"].add(metal)
    
    # Convert sets to strings
    mof_data = []
    for mof_uri, data in mof_dict.items():
        data["space_group"] = "; ".join(sorted(data["space_groups"])) if data["space_groups"] else "N/A"
        data["linkers"] = "; ".join(sorted(data["linkers"])) if data["linkers"] else "N/A"
        data["metal_node"] = "; ".join(sorted(data["metals"])) if data["metals"] else "N/A"
        mof_data.append(data)
    
    del g
    gc.collect()
    
    logging.info(f"Found {len(mof_data)} unique MOFs.")
    return mof_data

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load full KG from TTL
    nx_graph = load_kg_from_ttl(KG_PATH)
    
    # Extract MOF metadata
    mof_data = extract_mof_metadata(KG_PATH)

    # Convert to PyG
    logging.info("Converting NetworkX graph to PyTorch Geometric format...")
    nodes = list(nx_graph.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    
    # Build edge index
    edge_list = [[node_map[u], node_map[v]] for u, v in nx_graph.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Create PyG data object - create tensors on device from the start
    data = Data(edge_index=edge_index.to(device), num_nodes=len(nodes))
    data.x = torch.arange(len(nodes), dtype=torch.long, device=device)
    
    # Laplacian PE - compute on CPU first (can be memory intensive)
    pe_dim = 8
    logging.info("Computing Laplacian positional encodings...")
    # Temporarily move to CPU for PE computation
    data_cpu = data.cpu()
    transform = T.AddLaplacianEigenvectorPE(k=pe_dim, attr_name='pe', is_undirected=True)
    data_cpu = transform(data_cpu)
    if data_cpu.pe.shape[1] < pe_dim:
        padding = torch.zeros((data_cpu.num_nodes, pe_dim - data_cpu.pe.shape[1]))
        data_cpu.pe = torch.cat([data_cpu.pe, padding], dim=1)
    
    # Move PE back to device
    data.pe = data_cpu.pe.to(device)
    del data_cpu
    gc.collect()

    # Initialize Model with BigBird attention
    channels = 32
    model = GPSModel(num_nodes=len(nodes), channels=channels, pe_dim=pe_dim, use_bigbird=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    logging.info("Training Unsupervised GraphGPS Model with BigBird attention...")
    train(model, data, optimizer, epochs=100, device=device)

    # Extract Embeddings
    logging.info("Extracting embeddings...")
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.pe, data.edge_index).cpu().numpy()

    embeddings_list = []
    for mof in tqdm(mof_data, desc="Creating embedding rows"):
        mof_uri = mof["mof_uri"]
        if str(mof_uri) in node_map:
            idx = node_map[str(mof_uri)]
            embedding = z[idx]
            row = mof.copy()
            for i, val in enumerate(embedding):
                row[f"emb_{i}"] = val
            embeddings_list.append(row)

    df = pd.DataFrame(embeddings_list)
    logging.info(f"Saving GPS embeddings for {len(df)} MOFs to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info("Done.")

if __name__ == "__main__":
    main()