import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomTreesEmbedding, RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import torch as th
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

def data_split(X, y, train_ratio, val_ratio, seed=0):
    assert train_ratio + val_ratio < 1.0 
    N = X.shape[0]
    N_train = int(N*train_ratio)
    N_valid = int(N*val_ratio)
    
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(N)

    Xs = dict(
        train=th.tensor(X[shuffled_idx][:N_train]).float(),
        valid=th.tensor(X[shuffled_idx][N_train:N_train+N_valid]).float(),
        test=th.tensor(X[shuffled_idx][N_train+N_valid:]).float(),
    )
    ys = dict(
        train=th.tensor(y[shuffled_idx][:N_train]).float(),
        valid=th.tensor(y[shuffled_idx][N_train:N_train+N_valid]).float(),
        test=th.tensor(y[shuffled_idx][N_train+N_valid:]).float(),
    )
    return Xs, ys

def decision_tree_encoding(Xs, ys, max_splits=None, min_impurity_decrease=0.0, mode='classification'):
    dts = list()
    les = list()
    X_train_dt = th.zeros_like(Xs['train']).long()
    X_valid_dt = th.zeros_like(Xs['valid']).long()
    X_test_dt = th.zeros_like(Xs['test']).long()
    for idx in range(Xs['train'].shape[1]):
        if mode=='classification':
            dtc = DecisionTreeClassifier(
                max_leaf_nodes=max_splits, 
                min_impurity_decrease=min_impurity_decrease
            )
        elif mode=='regression':
            dtc = DecisionTreeRegressor(
                max_leaf_nodes=max_splits, 
                min_impurity_decrease=min_impurity_decrease
            )
        elif mode=='random':
            dtc = RandomTreesEmbedding(
                n_estimators=1,
                max_depth=12,
                max_leaf_nodes=max_splits, 
                min_impurity_decrease=min_impurity_decrease
            )
        dtc.fit(Xs['train'][:,idx].reshape((-1,1)), ys['train'])
        dts.append(dtc)
        le = LabelEncoder()
        X_train_dt[:,idx] = th.tensor(
            le.fit_transform(
                dtc.apply(Xs['train'][:,idx].reshape((-1,1))).reshape((-1,))
            )
        ).long()
        X_valid_dt[:,idx] = th.tensor(
            le.transform(
                dtc.apply(Xs['valid'][:,idx].reshape((-1,1))).reshape((-1,))
            )
        ).long()
        X_test_dt[:,idx] = th.tensor(
            le.transform(
                dtc.apply(Xs['test'][:,idx].reshape((-1,1))).reshape((-1,))
            )
        ).long()
        les.append(le)
    
    assert Xs['train'].shape==X_train_dt.shape
    Xs_dt = dict(
        train=X_train_dt,
        valid=X_valid_dt,
        test=X_test_dt,
    )
    return Xs_dt, dts, les

def decision_tree_encoding_multi(
        Xs, ys, tree_kwargs, mode='classification',
    ):
    
    dts = list()
    les = list()
    X_train_dt = th.zeros_like(Xs['train']).long()
    X_valid_dt = th.zeros_like(Xs['valid']).long()
    X_test_dt = th.zeros_like(Xs['test']).long()
    if mode=='classification':
        # dtc = GradientBoostingClassifier(**tree_kwargs)
        dtc = RandomForestClassifier(**tree_kwargs)
    elif mode=='regression':
#         dtc = RandomForestRegressor(**tree_kwargs)
        dtc = GradientBoostingRegressor(**tree_kwargs)
    elif mode=='random':
        dtc = RandomTreesEmbedding(**tree_kwargs)
    dtc.fit(Xs['train'], ys['train'])
    n_estimators = len(dtc.estimators_)
    dts.append(dtc)
    le = OrdinalEncoder()
    X_train_dt = th.tensor(
        le.fit_transform(
            dtc.apply(Xs['train']).reshape((-1, n_estimators))
        )
    ).long()
    X_valid_dt = th.tensor(
        le.transform(
            dtc.apply(Xs['valid']).reshape((-1, n_estimators))
        )
    ).long()
    X_test_dt = th.tensor(
        le.transform(
            dtc.apply(Xs['test']).reshape((-1, n_estimators))
        )
    ).long()
    les.append(le)
    
    Xs_dt = dict(
        train=X_train_dt,
        valid=X_valid_dt,
        test=X_test_dt,
    )
    return Xs_dt, dts, les

def train(
        model, X_train, y_train, X_valid, y_valid, X_test, 
        epochs=200, lr=0.01, stop_criteria=5, l1_lambda=0.0, embed_smooth_lambda=0.0,
    ):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5, verbose=True)
    train_losses = list()
    valid_losses = list()
    min_test_ = 1e8
    es_counter = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits.flatten(), y_train.float())
        loss_l1 = 0
        embed_smooth = 0
        if hasattr(model, "__getitem__") and isinstance(model[0], NumericEmbedding):
            for name, parm in model[0].named_parameters():
                if 'embed' in name:
                    loss_l1 += th.sum(th.abs(parm))
                    embed_smooth += th.sum((parm[1:,:]-parm[:-1,:])**2)
            loss += l1_lambda*loss_l1+embed_smooth_lambda*embed_smooth
        loss.backward()
        opt.step()
        
        train_losses.append(loss.item())
        model.eval()
        with th.no_grad():
            y_hat_valid = model(X_valid).flatten()
            valid_loss = F.binary_cross_entropy_with_logits(y_hat_valid, y_valid.float())
            valid_loss += l1_lambda*loss_l1 + embed_smooth_lambda*embed_smooth
            # scheduler.step(valid_loss)
            valid_losses.append(
                valid_loss.item()
            )
            valid_acc = ((y_hat_valid>=0.5).int()==y_valid.int()).float().mean().item()
            # if valid_losses[-1]<min_test_:
            #     min_test_ = valid_losses[-1]
            if -valid_acc < min_test_:
                min_test_ = -valid_acc
                es_counter = 0
            else:
                es_counter += 1
                
        if es_counter > stop_criteria:
            print(f"Early stopping at epoch {e} with valid acc {valid_acc:.3f}")
            break
    model.eval()
    return train_losses, valid_losses, model(X_test).flatten().sigmoid().detach().numpy()

class NumericEmbedding(nn.Module):
    def __init__(
        self, n_uniques, h_dim, agg_type='concat', 
        residual_method=None
        ):

        super().__init__()
        
        self.embeds = nn.ModuleList()
        for n_unique in n_uniques:
            self.embeds.append(
                nn.Embedding(n_unique, h_dim)
            )
        assert agg_type in ['concat', 'sum', 'mean']
        self.agg_type = agg_type
        assert residual_method in [None, 'concat', 'scale-add', 'scale-mult']
        self.residual_method = residual_method
        if self.residual_method in ['scale-add', 'scale-mult']:
            self.residual_eps = nn.Parameter(th.ones(len(n_uniques)).float())
    
    def forward(self, X):
        hs = list()
        for idx, embed in enumerate(self.embeds):
            hs.append(embed(X[:,idx].long()))

        if self.agg_type=='concat':
            h =  th.hstack(hs)
        elif self.agg_type=='sum':
            h =  th.stack(hs, dim=2).sum(dim=2).squeeze()
        elif self.agg_type=='mean':
            h =  th.stack(hs, dim=2).mean(dim=2).squeeze()
        
        if self.residual_method=='concat':
            h = th.cat([h, X[:,-1].float().reshape((-1,1))], dim=1)
        elif self.residual_method=='scale-add':
            # h = h + residual * parameters
            h = h + self.residual_eps * X[:,-1].float().reshape((-1,1))
        elif self.residual_method=='scale-mult':
            # h = h * residual * parameters
            h = h * self.residual_eps * X[:,-1].float().reshape((-1,1))

        return h

class FeedForwardMLPResidual(nn.Module):
    def __init__(
        self, d_model, dim_feedforward, dropout, 
        layer_norm_eps=1e-5, norm_first=False, activation=F.relu,
        ):
        super().__init__()
        self.activation = activation
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    
    def forward(self, src):
        x = src
        if self.norm_first:
            x = x + self._ff_block(self.norm(x))
        else:
            x = self.norm(x + self._ff_block(x))
        return x
