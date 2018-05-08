from PhysicsTools.NanoAOD.common_cff import *
import FWCore.ParameterSet.Config as cms
#diMuCand = cms.EDProducer("CandViewShallowCloneCombiner", 
#  decay = cms.string('slimmedMuons@+ slimmedMuons@-'),cut = cms.string(''))

BToKmumu=cms.EDProducer("BToKmumuProducer", 
                        src=cms.InputTag("slimmedMuons"))
BToKmumuTable=cms.EDProducer("SimpleCompositeCandidateFlatTableProducer", 
                             src=cms.InputTag("BToKmumu"),
                             cut=cms.string(""),
                             name=cms.string("BToKmumu"),
                             doc=cms.string("BToKmumu Variable"),
                             singleton=cms.bool(False),
                             extension=cms.bool(False),
                             variables=cms.PSet(Lxy=Var("userFloat('Lxy')", float,doc="Lxy")))
BToKmumuSequence=cms.Sequence(#diMuCand+
                              BToKmumu)
BToKmumuTableSequence=cms.Sequence(BToKmumuTable)

