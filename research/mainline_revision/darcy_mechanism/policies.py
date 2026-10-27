import bootstrap
import numpy as np

def teacher_mean(teacher, ids):
    coef = np.einsum('bd,edr->ebr', teacher.features[ids], teacher.weights)
    return coef.mean(0) @ teacher.basis.T

def field_score(teacher, ids):
    coef = np.einsum('bd,edr->ebr', teacher.features[ids], teacher.weights)
    mean = coef.mean(0)
    # The teacher basis is orthonormal; this is exactly the mean field variance.
    q = np.mean(np.sum((coef-mean[None])**2, axis=-1), axis=0)/teacher.basis.shape[0]
    q += float(np.mean(teacher.audit_residual**2))*1e-4
    return mean @ teacher.basis.T, q
