"""
Created on October 18, 2018

@author: Nicola Di Mauro
"""
from Inference.stats.Moments import add_node_moment
from Structure.leaves.cltree.CLTree import *
import logging

logger = logging.getLogger(__name__)


def cltree_expectation(node, order=1):
    raise ValueError("Not Implemented")


def add_cltree_expectation_support():
    add_node_moment(CLTree, cltree_expectation)
