import torch
import GMN.utils as gmnutils
import torch.nn.functional as F
import GMN.graphembeddingnetwork as gmngen
from utils import model_utils
import torch_geometric as pyg


class GEN(torch.nn.Module):
    def __init__(self,conf, gmn_config):
        super(GEN, self).__init__()
        self.config = gmn_config
        self.build_layers()

        self.diagnostic_mode = False
        self.fetch_embed = False
        self.device = conf.training.device
        assert conf.model.scoring_function in model_utils.scoring_functions.keys()
        self.scoring_function = model_utils.scoring_functions[conf.model.scoring_function]
        self.use_sig = False
        if conf.model.scoring_function == "sighinge":
            self.use_sig = True
            self.sigmoid_a = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))
            self.sigmoid_b = torch.nn.Parameter(torch.tensor(1,dtype=torch.float32))        
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"])
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"])


    def forward(self, batch_data, batch_data_node_sizes, batch_data_edge_sizes):
        """
        """
        node_features, edge_features, from_idx, to_idx, graph_idx = model_utils.get_graph_features(batch_data)

        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for i in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx,edge_features_enc)
            
        graph_vectors = self.aggregator(node_features_enc,graph_idx, len(batch_data_node_sizes) )
        x, y = gmnutils.reshape_and_split_tensor(graph_vectors, 2)
        if self.fetch_embed:
            return graph_vectors
        if self.use_sig:
            scores = self.scoring_function(self.sigmoid_a, self.sigmoid_b, x, y)
        else:
            scores = self.scoring_function(x,y)
        return scores


class NANL(torch.nn.Module):
    def __init__(self,conf, gmn_config):
        super(NANL, self).__init__()
        self.config = gmn_config
        self.diagnostic_mode = False
        self.fetch_embed = False
        self.device = conf.training.device
        self.max_node_set_size = conf.actual_max_node_set_size
        self.build_layers()

        self.graph_size_to_mask_map = model_utils.graph_size_to_mask_map(
            max_set_size=self.max_node_set_size, lateral_dim=self.max_node_set_size, device=self.device
        )
        self.sinkhorn_temp = conf.training.sinkhorn_temp
        self.scoring_layer = conf.dataset.rel_mode
        
    def build_layers(self):
        self.encoder = gmngen.GraphEncoder(**self.config["encoder"]).to(self.device)
        prop_config = self.config["graph_embedding_net"].copy()
        prop_config.pop("n_prop_layers", None)
        prop_config.pop("share_prop_params", None)
        self.prop_layer = gmngen.GraphPropLayer(**prop_config).to(self.device)
        self.aggregator = gmngen.GraphAggregator(**self.config["aggregator"]).to(self.device)
        self.node_sinkhorn_feature_layers = torch.nn.Sequential(
            torch.nn.Linear(prop_config["node_state_dim"], self.max_node_set_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.max_node_set_size, self.max_node_set_size),
        ).to(self.device)

    def get_transport_plan(self, stacked_node_features_query, stacked_node_features_corpus, query_sizes, corpus_sizes):
        
        
        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_node_features_query)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_node_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        node_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input, device=self.device,temperature=self.sinkhorn_temp 
        )
    
        
        return  node_transport_plan

    def forward(self, graphs, graph_node_sizes, graph_edge_sizes):
        query_sizes = graph_node_sizes[0::2]
        corpus_sizes = graph_node_sizes[1::2]


        node_features, edge_features, from_idx, to_idx, graph_idx= model_utils.get_graph_features(graphs)

        # Propagation to compute node embeddings
        node_features_enc, edge_features_enc = self.encoder(node_features, edge_features)
        for _ in range(self.config["graph_embedding_net"]["n_prop_layers"]):
            node_features_enc = self.prop_layer(node_features_enc, from_idx, to_idx, edge_features_enc)

        if self.fetch_embed:
            return model_utils.split_and_stack_singles(node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size)

        stacked_node_features_query, stacked_node_features_corpus = model_utils.split_and_stack_pairs(
            node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size
        )

        # Computation of node transport plan
        transformed_features_query = self.node_sinkhorn_feature_layers(stacked_node_features_query)
        transformed_features_corpus = self.node_sinkhorn_feature_layers(stacked_node_features_corpus)

        def mask_graphs(features, graph_sizes):
            mask = torch.stack([self.graph_size_to_mask_map[i] for i in graph_sizes])
            return mask * features
        masked_features_query = mask_graphs(transformed_features_query, query_sizes)
        masked_features_corpus = mask_graphs(transformed_features_corpus, corpus_sizes)

        node_sinkhorn_input = torch.matmul(masked_features_query, masked_features_corpus.permute(0, 2, 1))
        node_transport_plan = model_utils.pytorch_sinkhorn_iters(
            log_alpha=node_sinkhorn_input, device=self.device,temperature=self.sinkhorn_temp 
        )
    
        if self.diagnostic_mode:
            return node_features_enc, node_transport_plan, stacked_node_features_query, stacked_node_features_corpus, node_transport_plan
        elif self.fetch_embed:
            return model_utils.split_and_stack_singles(node_features_enc, graph_node_sizes.tolist(), self.max_node_set_size)
        else:
            if self.scoring_layer == "sub_iso":
                return model_utils.subiso_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
            elif self.scoring_layer == "ged":
                return model_utils.ged_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
            elif self.scoring_layer == "uneq_ged":
                return model_utils.uneq_ged_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
            elif self.scoring_layer == "mcs":
                # return model_utils.mcs_feature_alignment_score(stacked_node_features_query, stacked_node_features_corpus, node_transport_plan)
                raise NotImplementedError("MCS not implemented")
            else:
                raise NotImplementedError(f"Scoring layer {self.scoring_layer} not implemented")
            
            
