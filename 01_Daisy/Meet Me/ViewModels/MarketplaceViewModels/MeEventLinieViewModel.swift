//
//  MeEventLinieViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 18.02.21.
//

import Foundation
import PromiseKit
import Firebase

class MeEventLineViewModel: ObservableObject {
    
    @Published var eventArray: [EventModel] = []
    private var firestoreManagerEventTest = FirestoreManagerEventTest()
    private var firestoreFotoManagerEventTest: FirestoreFotoManagerEventTest = FirestoreFotoManagerEventTest()
    private var db: Firestore
    
    init() {
        db = Firestore.firestore()

    }
    
    
    func getMeEvents() {
        firstly {
            firebaseGetMeEvents()
        }.catch { error in
            print("DEBUG: error in getMeEvents error: \(error)")
            print("DEGUB: error localized \(error.localizedDescription)")
        }
    }
    
    
    // MARK: - Functions to get events
    func firebaseGetMeEvents() -> Promise<Void> {
        return Promise { seal in
            
            guard let currentUser = Auth.auth().currentUser else {
                let error = Err("No User Profil")
                seal.reject(error)
                return
            }
            
            
            db.collection("events")
                .whereField("userId", isEqualTo: currentUser.uid)
                .addSnapshotListener {(snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        
                        if let snapshot = snapshot {
                            let event: [EventModel]? = snapshot.documents.compactMap { doc in
                                var event = try? doc.data(as: EventModel.self)
                                event?.eventId = doc.documentID
                                if let event = event {
                                    return event
                                }
                                return nil
                                
                            }
                            DispatchQueue.main.async {
                                if let event = event {
                                    self.eventArray = event
                                    
                                    self.eventArray = self.eventArray.sorted {
                                        if $0.eventMatched && $1.eventMatched == false {
                                            return ($0.likedUser && !$1.likedUser)
                                        }
                                        return (!$0.eventMatched && $1.eventMatched)
                                    }
                                    seal.fulfill(())
                                }
                                
                            }
                            
                        }
                    }
                }
            
            
        }
    }
}

//self.matches = self.matches.sorted{
//    ($0.event.date,$0.event.startTime ) <
//        ($1.event.date,$1.event.startTime )
//
//}



