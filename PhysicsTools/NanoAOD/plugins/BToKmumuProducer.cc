#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include <DataFormats/PatCandidates/interface/CompositeCandidate.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/Utilities/interface/StreamID.h"
#include <FWCore/Framework/interface/MakerMacros.h>
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include <TLorentzVector.h>
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <FWCore/Framework/interface/ESHandle.h>
//
// class declaration
//

class BToKmumuProducer : public edm::EDProducer {
   
   public:

   explicit BToKmumuProducer(const edm::ParameterSet &iConfig);
  //  srcDiMuons_(consumes<edm::View<reco::CompositeCandidate>>(iConfig.getParameter<edm::InputTag>("srcDiMuon")))
  // {
  //  produces<pat::CompositeCandidateCollection>();
  // }
   ~BToKmumuProducer() override {};

   
   private:

   virtual void produce(edm::Event&, const edm::EventSetup&);

   bool hasBeamSpot(const edm::Event&);

   bool hasGoodMuMuVertex (const reco::TransientTrack, const reco::TransientTrack,
			   reco::TransientTrack &, reco::TransientTrack &,
			   double &, double &, double &, double &, double &,
			   double &, double &, double &);

   void computeLS (double, double, double, double, double, double, double,
		   double, double, double, double, double, double, double,
		   double, double, double, double, double*, double*);

   void computeCosAlpha (double, double, double, double, double,
			 double, double, double, double, double,
			 double, double, double, double,
			 double, double, double, double,double*, double*);
  
  // ----------member data ---------------------------

  edm::EDGetTokenT<std::vector<pat::Muon>> src_;

  //edm::EDGetTokenT<edm::View<reco::CompositeCandidate>> srcDiMuons_;

  edm::ESHandle<MagneticField> bFieldHandle_;

  edm::InputTag BeamSpotLabel_;

  reco::BeamSpot beamSpot_;

};



BToKmumuProducer::BToKmumuProducer(const edm::ParameterSet &iConfig):
  src_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src")))
   {
     produces<pat::CompositeCandidateCollection>();
   }


void BToKmumuProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<std::vector<pat::Muon>> src; 

  iEvent.getByToken(src_, src);

  unsigned int muonNumber = src->size();

  // Output collection 
  std::unique_ptr<pat::CompositeCandidateCollection> result( new pat::CompositeCandidateCollection );

  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle_);
  
  std::cout << "ciao1\n";

  if(muonNumber>1){
    
  // loop on all the pairs 
    for (unsigned int i = 0; i < muonNumber-1; ++i) {

      const pat::Muon & muon1 = (*src)[i];

      if(!muon1.isLooseMuon())continue;

      for (unsigned int j = i+1; j < muonNumber; ++j) {
	
	const pat::Muon & muon2 = (*src)[j];

	if(!muon2.isLooseMuon())continue;

	//Get the pair and the two leptons composing it 
	//const reco::CompositeCandidate& pairBuf = (*srcDiMuons)[i]; 
	//pat::CompositeCandidate   pair(pairBuf); 

	std::cout << "ciao2\n";

	//const reco::Candidate *l1 = pair.daughter(0);
	//const reco::Candidate *l2 =    pair.daughter(1);
	//std::cout<<l1->pt()<<"\n";
	//std::cout<<l2->pt()<<"\n";

	//pat::MuonRef muon1, muon2;

	//if ( l1->hasMasterClone() ) { muon1 = l1->masterClone().castTo<pat::MuonRef>(); }
	//if ( l2->hasMasterClone() ) { muon2 = l2->masterClone().castTo<pat::MuonRef>(); }
	//if (l1->charge()*l2->charge()>0)continue;

	std::cout << "ciao3\n";

	std::cout << muon1.innerTrack().isNull()<<" "<< muon2.innerTrack().isNull()<<"\n";

	reco::TrackRef muTrackp = muon1.charge()>0 ? muon1.innerTrack():muon2.innerTrack();

	std::cout << "ciao4\n";

	reco::TrackRef muTrackm = muon1.charge()<0 ? muon1.innerTrack():muon2.innerTrack();

	std::cout << muTrackp.isNull();
   
	//std::cout << muTrackm.isNull();
   
	std::cout << "ciao5\n";

	const reco::TransientTrack muTrackpTT(muTrackp, &(*bFieldHandle_));

	std::cout << "ciao6\n";
   
        const reco::TransientTrack muTrackmTT(muTrackm, &(*bFieldHandle_));

	std::cout << "ciao7\n";

        reco::TransientTrack refitMupTT, refitMumTT; 
	double mu_mu_vtx_cl, mu_mu_pt, mu_mu_mass, mu_mu_mass_err; 
	double MuMuLSBS, MuMuLSBSErr; 
	double MuMuCosAlphaBS, MuMuCosAlphaBSErr;
 
	bool passed;
	passed = hasGoodMuMuVertex(muTrackpTT, muTrackmTT, refitMupTT, refitMumTT,
				   mu_mu_vtx_cl, mu_mu_pt, mu_mu_mass, 
				   mu_mu_mass_err, MuMuLSBS, MuMuLSBSErr, 
				   MuMuCosAlphaBS, MuMuCosAlphaBSErr);

	std::cout << "passed "<< passed << "\n";

	if ( !passed) continue;

	pat::CompositeCandidate pair;
	pair.addDaughter( muon1 );
	pair.addDaughter( muon2 );
   
	pair.addUserFloat("Lxy", (float) MuMuLSBS/MuMuLSBSErr);

	std::cout <<"\n"<<  MuMuLSBS/MuMuLSBSErr<<"\n";

	result->push_back(pair);
   
      }

    }

  }
 
  //iEvent.put(result);
  iEvent.put(std::move(result));   
  
}




bool
BToKmumuProducer::hasBeamSpot(const edm::Event& iEvent)
{

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel(BeamSpotLabel_, beamSpotHandle);
  
  if ( ! beamSpotHandle.isValid() ) {
    edm::LogError("myBeam") << "No beam spot available from EventSetup" ;
    return false;
  }
  
  beamSpot_ = *beamSpotHandle;
  return true;
}



bool 
BToKmumuProducer::hasGoodMuMuVertex (const reco::TransientTrack muTrackpTT,
				     const reco::TransientTrack muTrackmTT,
				     reco::TransientTrack &refitMupTT,
				     reco::TransientTrack &refitMumTT,
				     double & mu_mu_vtx_cl, double & mu_mu_pt,
				     double & mu_mu_mass, double & mu_mu_mass_err,
				     double & MuMuLSBS, double & MuMuLSBSErr,
				     double & MuMuCosAlphaBS,
				     double & MuMuCosAlphaBSErr){
  //io
  //double MuMuMinInvMass_;
  //double MuMuMaxInvMass_;
  //double MuMuMinLxySigmaBs_;
  //double MuMuMinCosAlphaBs_;
  //double MuMuMinVtxCl_;
  //double MuMuMinPt_;
  ParticleMass MuonMass_;
  float MuonMassErr_;

  KinematicParticleFactoryFromTransientTrack partFactory;
  KinematicParticleVertexFitter PartVtxFitter;

  std::vector<RefCountedKinematicParticle> muonParticles;
  double chi = 0.;
  double ndf = 0.;
  muonParticles.push_back(partFactory.particle(muTrackmTT,
					       MuonMass_,chi,ndf,MuonMassErr_));
  muonParticles.push_back(partFactory.particle(muTrackpTT,MuonMass_,chi,ndf,MuonMassErr_));
  RefCountedKinematicTree mumuVertexFitTree = PartVtxFitter.fit(muonParticles);
  
  if ( !mumuVertexFitTree->isValid()) return false;
  
  std::cout << mumuVertexFitTree->isValid();

  mumuVertexFitTree->movePointerToTheTop();
  RefCountedKinematicParticle mumu_KP = mumuVertexFitTree->currentParticle();
  RefCountedKinematicVertex mumu_KV = mumuVertexFitTree->currentDecayVertex();
  
  if ( !mumu_KV->vertexIsValid()) return false;
  
  mu_mu_vtx_cl = TMath::Prob((double)mumu_KV->chiSquared(),
			     int(rint(mumu_KV->degreesOfFreedom())));
  
  //if (mu_mu_vtx_cl < MuMuMinVtxCl_) return false;

  // extract the re-fitted tracks
  mumuVertexFitTree->movePointerToTheTop();
  
  mumuVertexFitTree->movePointerToTheFirstChild();
  RefCountedKinematicParticle refitMum = mumuVertexFitTree->currentParticle();
  refitMumTT = refitMum->refittedTransientTrack();
  
  mumuVertexFitTree->movePointerToTheNextChild();
  RefCountedKinematicParticle refitMup = mumuVertexFitTree->currentParticle();
  refitMupTT = refitMup->refittedTransientTrack();
  
  TLorentzVector mymum, mymup, mydimu;
  
  mymum.SetXYZM(refitMumTT.track().momentum().x(),
                refitMumTT.track().momentum().y(),
                refitMumTT.track().momentum().z(), MuonMass_);

  mymup.SetXYZM(refitMupTT.track().momentum().x(),
                refitMupTT.track().momentum().y(),
                refitMupTT.track().momentum().z(), MuonMass_);
  
  mydimu = mymum + mymup;
  mu_mu_pt = mydimu.Perp();
 
  mu_mu_mass = mumu_KP->currentState().mass();
  mu_mu_mass_err = sqrt(mumu_KP->currentState().kinematicParametersError().
                        matrix()(6,6));

  //if ((mu_mu_pt < MuMuMinPt_) || (mu_mu_mass < MuMuMinInvMass_) ||
  //    (mu_mu_mass > MuMuMaxInvMass_)) return false;

  // compute the distance between mumu vtx and beam spot
  computeLS (mumu_KV->position().x(),mumu_KV->position().y(),0.0,
	     beamSpot_.position().x(),beamSpot_.position().y(),0.0,
	     mumu_KV->error().cxx(),mumu_KV->error().cyy(),0.0,
	     mumu_KV->error().matrix()(0,1),0.0,0.0,
	     beamSpot_.covariance()(0,0),beamSpot_.covariance()(1,1),0.0,
	     beamSpot_.covariance()(0,1),0.0,0.0,
	     &MuMuLSBS,&MuMuLSBSErr);
  
  //if (MuMuLSBS/MuMuLSBSErr < MuMuMinLxySigmaBs_) return false;

  computeCosAlpha(mumu_KP->currentState().globalMomentum().x(),
		  mumu_KP->currentState().globalMomentum().y(),
		  0.0,
		  mumu_KV->position().x() - beamSpot_.position().x(),
		  mumu_KV->position().y() - beamSpot_.position().y(),
		  0.0,
		  mumu_KP->currentState().kinematicParametersError().matrix()(3,3),
		  mumu_KP->currentState().kinematicParametersError().matrix()(4,4),
		  0.0,
		  mumu_KP->currentState().kinematicParametersError().matrix()(3,4),
		  0.0,
		  0.0,
		  mumu_KV->error().cxx() + beamSpot_.covariance()(0,0),
		  mumu_KV->error().cyy() + beamSpot_.covariance()(1,1),
		  0.0,
		  mumu_KV->error().matrix()(0,1) + beamSpot_.covariance()(0,1),
		  0.0,
		  0.0,
		  &MuMuCosAlphaBS,&MuMuCosAlphaBSErr);        
  
  //if (MuMuCosAlphaBS < MuMuMinCosAlphaBs_) return false;

  return true;
}



void 
BToKmumuProducer::computeLS (double Vx, double Vy, double Vz,
			     double Wx, double Wy, double Wz,
			     double VxErr2, double VyErr2, double VzErr2,
			     double VxyCov, double VxzCov, double VyzCov,
			     double WxErr2, double WyErr2, double WzErr2,
			     double WxyCov, double WxzCov, double WyzCov,
			     double* deltaD, double* deltaDErr){

  *deltaD = sqrt((Vx-Wx) * (Vx-Wx) + (Vy-Wy) * (Vy-Wy) + (Vz-Wz) * (Vz-Wz));
  if (*deltaD > 0.)
    *deltaDErr = sqrt((Vx-Wx) * (Vx-Wx) * VxErr2 +
		      (Vy-Wy) * (Vy-Wy) * VyErr2 +
		      (Vz-Wz) * (Vz-Wz) * VzErr2 +
		      
		      (Vx-Wx) * (Vy-Wy) * 2.*VxyCov +
		      (Vx-Wx) * (Vz-Wz) * 2.*VxzCov +
		      (Vy-Wy) * (Vz-Wz) * 2.*VyzCov +
                
		      (Vx-Wx) * (Vx-Wx) * WxErr2 +
		      (Vy-Wy) * (Vy-Wy) * WyErr2 +
		      (Vz-Wz) * (Vz-Wz) * WzErr2 +
                
		      (Vx-Wx) * (Vy-Wy) * 2.*WxyCov +
		      (Vx-Wx) * (Vz-Wz) * 2.*WxzCov +
		      (Vy-Wy) * (Vz-Wz) * 2.*WyzCov) / *deltaD;
  else *deltaDErr = 0.;
}





void 
BToKmumuProducer::computeCosAlpha (double Vx, double Vy, double Vz,
				   double Wx, double Wy, double Wz,
				   double VxErr2, double VyErr2, double VzErr2,
				   double VxyCov, double VxzCov, double VyzCov,
				   double WxErr2, double WyErr2, double WzErr2,
				   double WxyCov, double WxzCov, double WyzCov,
				   double* cosAlpha, double* cosAlphaErr){

  double Vnorm = sqrt(Vx*Vx + Vy*Vy + Vz*Vz);
  double Wnorm = sqrt(Wx*Wx + Wy*Wy + Wz*Wz);
  double VdotW = Vx*Wx + Vy*Wy + Vz*Wz;
  
  if ((Vnorm > 0.) && (Wnorm > 0.)) {
    *cosAlpha = VdotW / (Vnorm * Wnorm);
    *cosAlphaErr = sqrt( (
			  (Vx*Wnorm - VdotW*Wx) * (Vx*Wnorm - VdotW*Wx) * WxErr2 +
			  (Vy*Wnorm - VdotW*Wy) * (Vy*Wnorm - VdotW*Wy) * WyErr2 +
			  (Vz*Wnorm - VdotW*Wz) * (Vz*Wnorm - VdotW*Wz) * WzErr2 +
       
			  (Vx*Wnorm - VdotW*Wx) * (Vy*Wnorm - VdotW*Wy) * 2.*WxyCov +
			  (Vx*Wnorm - VdotW*Wx) * (Vz*Wnorm - VdotW*Wz) * 2.*WxzCov +
			  (Vy*Wnorm - VdotW*Wy) * (Vz*Wnorm - VdotW*Wz) * 2.*WyzCov) /
                         (Wnorm*Wnorm*Wnorm*Wnorm) +
			 
                         ((Wx*Vnorm - VdotW*Vx) * (Wx*Vnorm - VdotW*Vx) * VxErr2 +
			  (Wy*Vnorm - VdotW*Vy) * (Wy*Vnorm - VdotW*Vy) * VyErr2 +
			  (Wz*Vnorm - VdotW*Vz) * (Wz*Vnorm - VdotW*Vz) * VzErr2 +
                        
			  (Wx*Vnorm - VdotW*Vx) * (Wy*Vnorm - VdotW*Vy) * 2.*VxyCov +
			  (Wx*Vnorm - VdotW*Vx) * (Wz*Vnorm - VdotW*Vz) * 2.*VxzCov +
			  (Wy*Vnorm - VdotW*Vy) * (Wz*Vnorm - VdotW*Vz) * 2.*VyzCov) /
                         (Vnorm*Vnorm*Vnorm*Vnorm) ) / (Wnorm*Vnorm);
  } else {
    *cosAlpha = 0.;
    *cosAlphaErr = 0.;
  }
}


DEFINE_FWK_MODULE(BToKmumuProducer);
