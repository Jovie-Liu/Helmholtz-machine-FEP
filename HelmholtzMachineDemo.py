# HelmholtzMachineDemo.py
# 6/12/2015   
# Kevin Kirby, Northern Kentucky University
"""
A basic Python implementation/demonstration of a Helmholtz Machine.
A companion to my 2006 tutorial.

@author: Kevin Kirby
"""

import numpy as np



# Some utility functions

def sigmoid(u):
    """
    A vectorized version of the logistic sigmoid firing function.
    """
    return 1/(1 + np.exp(np.negative(u))) 
    
def bernoulliSample( probs ):
    """
    Generate a binary vector according to a vector of probabilities.
    """
    return map( (lambda p : np.random.choice( (1.0,0.0), p=(p,1-p) )), probs )
    
def intToPattern( i, n ):
    """
    An length n tuple of 0.,1.'s corresponding to the binary form of i.
    """
    return tuple( map(float, bin(i)[2:].zfill(n)))        
    
def patternToString( d, clumpsize ):
    """
    For prettyprinting a real binary pattern list in space-separated clumps.
    """
    s= ""
    for k in range( len(d) ):
        if k%clumpsize == 0:
            s+= " "
        s += "1" if d[k]>0 else "0"
    return s   
    
    

# An toy example of a world of tiny fluctuating images. 
    
vhWorld={
    # Vertically striped 3x3 images occur with probability 2/18 each.
    (1,0,0, 1,0,0, 1,0,0) : 2/18.,
    (0,1,0, 0,1,0, 0,1,0) : 2/18.,  
    (0,0,1, 0,0,1, 0,0,1) : 2/18.,    
    (1,1,0, 1,1,0, 1,1,0) : 2/18.,  
    (1,0,1, 1,0,1, 1,0,1) : 2/18.,    
    (0,1,1, 0,1,1, 0,1,1) : 2/18., 
    # Horizontally striped 3x3 images occur with probability 1/18 each.
    (1,1,1, 0,0,0, 0,0,0) : 1/18.,
    (0,0,0, 1,1,1, 0,0,0) : 1/18.,
    (0,0,0, 0,0,0, 1,1,1) : 1/18.,
    (1,1,1, 1,1,1, 0,0,0) : 1/18.,
    (1,1,1, 0,0,0, 1,1,1) : 1/18.,
    (0,0,0, 1,1,1, 1,1,1) : 1/18.
    # The remaining 500 (of 512 possible) 3x3 images occur with probablility 0.
    }


def sampleFromWorld( world ):
    """
    Return a random pattern from world according to its associated probability.
    """
    index= np.random.choice( len( world ), p=world.values() )
    return list( world.keys()[index] ) 
      


# A basic Helmholtz machine class.
  
class HelmholtzMachine:
    def __init__( self, nTop, nMid, nBottom ):
        """
        Construct a 3-layer HM with the given number of neurons at each level.
        """
        # Dimension of pattern vector visible at the bottom.
        self.dimension= nBottom
        
        # Generative weights (top down)
        self.bG= np.zeros( nTop )
        self.wG= np.zeros( [nMid, nTop+1] )
        self.vG= np.zeros( [nBottom, nMid+1] )
        
        # Recognition weights (bottom up)
        self.vR= np.zeros( [nMid, nBottom+1] )        
        self.wR= np.zeros( [nTop, nMid+1] )
        
        # Some default learning rates
        self.eps= 0.01       # default learning rate 
        self.epsBottom= 0.15 # special learning rate for bottom layer
        
    def generate( self ):
        """
        Generate (fantasize) a pattern d.
        """
        x= bernoulliSample( sigmoid( self.bG ) )
        y= bernoulliSample( sigmoid( np.dot(self.wG, x+[1]) ) )
        d= bernoulliSample( sigmoid( np.dot(self.vG, y+[1]) ) )
        return d
      
    def wake( self, d ):
        """
        One wake learning cycle, given pattern d applied at the bottom.
        """
        # Upward (recognition) pass.
        y= bernoulliSample( sigmoid( np.dot(self.vR, d+[1]) ) )
        x= bernoulliSample( sigmoid( np.dot(self.wR, y+[1]) ) )
        
        # Downward (generation) pass.
        px= sigmoid( self.bG ) 
        py= sigmoid( np.dot(self.wG, x+[1]) ) 
        pd= sigmoid( np.dot(self.vG, y+[1]) ) 
        
        # Adjust generative weights by delta rule.
        self.bG += self.eps * ( x - px ) 
        self.wG += self.eps * np.outer( y - py, x+[1] ) 
        self.vG += self.epsBottom * np.outer( d - pd, y+[1] )
         
    def sleep( self ):
        """
        One sleep learning cycle.
        """
        # Initiate a dream!
        x= bernoulliSample( sigmoid( self.bG ) ) 
        
        # Pass dream signal downward to generate a pattern.
        y= bernoulliSample( sigmoid( np.dot(self.wG, x+[1]) ) )
        d= bernoulliSample( sigmoid( np.dot(self.vG, y+[1]) ) )
        
        # Pass back up through recognition network, saving computed probabilities.
        py= sigmoid( np.dot(self.vR, d+[1]) ) 
        px= sigmoid( np.dot(self.wR, y+[1]) )
        
        # Adjust recognition weights by delta rule. 
        self.vR += self.epsBottom * np.outer( y - py, d+[1] )
        self.wR += self.eps       * np.outer( x - px, y+[1] )
        
    def learn( self, world, nCycles ):
        """
        Run for a given number of wake-sleep cycles to learn this world.
        """
        for t in range( nCycles ):
            d= sampleFromWorld( world )
            self.wake( d )
            self.sleep()        
        
    def estimateModel( self, nSamples ):
        """
        Generate many samples in order to estimate pattern probabilities.
        probG[d] will be the estimated probability that the HM generates pattern d.
        """
        dp= 1.0 / nSamples
        probG= dict()  
        for count in range(nSamples):
            d= tuple( self.generate() )
            if d in probG:
                probG[ d ] += dp
            else:
                probG[ d ] = dp
        return probG


  
def reportModelComparison( world, machine ):
    """
    Print a table that shows how closely the machine models the world.
    """
    # Get table of (d,pG(d)) by doing a lot of sampling from the machine.
    probG= machine.estimateModel( 100000 )      
    
    # Loop through all the patterns in the world.
    dim= machine.dimension
    assert dim <= 16  # This report only good for "toy" (low-dimension) examples
    klSum= 0  # for accumulating KL divergence
    plist= [] # to hold a list (d,p(d), pG(d)) to print as a summary
    for i in range (2**dim):
        d= intToPattern(i,dim)
        p= world[d] if d in world else 0   # probablity the world shows d
        pG= probG[d] if d in probG else 0  # probability the machine generates d
        if (p > 0) and (pG > 0):
            klSum += p * np.log( p / pG )
        if (p > 0.001 ) or (pG > 0.001):
            plist += [(d,p,pG)] # only interested in reporting on non-miniscule probabilities
     
    # Sort the list of (d,p(d),pG(d)) by p(d) descending and print.
    list.sort( plist, key=lambda x: x[1], reverse=True )
    print "The world vs. the Helmholtz machine's current model of the world:"
    for (d,p,pG) in plist:
        print patternToString(d,3), "  %5.3f" % p, "  %5.3f" % pG
    print "KL(world,machine) = ", klSum        
        



# Demonstrate how a small vanilla Helmholtz machine can learn to model vhWorld.
# It's sensitive to randomness in sampling, but most of the time it can learn the
# world nicely in under 80,000 cycles.

hm= HelmholtzMachine( 1, 4, 9 )  
hm.learn( vhWorld, 80000 )       
reportModelComparison( vhWorld, hm )




    
