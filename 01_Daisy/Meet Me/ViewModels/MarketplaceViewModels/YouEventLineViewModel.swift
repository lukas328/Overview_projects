//
//  YouEventViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 16.02.21.
//

import Foundation
import PromiseKit
import MapKit
import GeoFire
import Firebase

class YouEventLineViewModel: ObservableObject {
    private var firestoreManagerEventTest: FirestoreManagerEventTest = FirestoreManagerEventTest()
    private var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()
    @Published var userModel: UserModel = stockUser
    private var db: Firestore
    @Published var refreshCounter = 0
    @Published var changedSearchingFor = false
    
    private var reportedEvents: [String] = []

  
    
    init() {
        db = Firestore.firestore()
        
    }
    


    func getYouEvents(region: MKCoordinateRegion, shuffle: Bool) -> Promise<[EventModel]> {
        return Promise { seal in
            firstly {
                firestoreManagerUserTest.getAllReportedEvents()
            }.map { reportedEvents in
                self.reportedEvents = reportedEvents
            }.then{
                 self.firestoreManagerUserTest.getAllLikedEvents()//, self.getLocation())
            }.then { likedEvents in
                self.firestoreManagerEventTest.queryColletion(center: region.center, user: self.userModel).map { ($0, likedEvents) }
            }.then { queries, likedEvents in
                self.firestoreManagerEventTest.querysInEvent(likedEvents: likedEvents , queries: queries, center: region.center, user: self.userModel, shuffle: shuffle, reportedEvents: self.reportedEvents)
            }.done { events in
                seal.fulfill(events)
            }.catch { error in
                seal.reject(error)
            }
        }
    }
    
    func getYouEventsTest(shuffle: Bool) -> Promise<[EventModel]> {
        return Promise { seal in
            firstly {
                self.firestoreManagerUserTest.getAllLikedEvents()
            }.then { likedEvent in
                self.firestoreManagerEventTest.firebaseGetYouEventsTest(likedEvents: likedEvent)
            }.done { events in
                seal.fulfill(events)
            }.catch { error in
                seal.reject(error)
            }
        }
    }
    
    func reportEvent(eventModel: EventModel) {
        firstly {
            self.firestoreManagerUserTest.addReportToReportArray(eventId: eventModel.eventId)
        }.then {
            self.firestoreManagerUserTest.addOneToReportCounter(userId: eventModel.userId)
        }.catch { error in
            print(error)
        }
    }
    
    func addOneToRefreshCounter() {
        firstly {
            self.firestoreManagerUserTest.addOneToRefreshCounter()
        }.catch { error in
            print("DEBUG: Error in refreshCounter, \(error)")
            print(error.localizedDescription)
        }
    }
    
    func getRefreshCounter() ->Promise<Void> {
        return Promise { seal in
        guard let currentUser = Auth.auth().currentUser else {
            return
        }
        db.collection("users").document(currentUser.uid).addSnapshotListener { snapshot, error in
                if let error = error {
                    print(error.localizedDescription)
                } else {
                    if let snapshot = snapshot {
                        let userModel = try? snapshot.data(as: UserModel.self)
                        DispatchQueue.main.async {
                            if userModel != nil {
                                if userModel!.searchingFor != self.userModel.searchingFor {
                                    self.changedSearchingFor = true
                                }
                                self.refreshCounter = userModel!.refreshCounter
                                self.userModel = userModel!
                                seal.fulfill(())
                            }else {
                                print("Error by getting refresh counter")
                                
                            }
                        }
                        
                        
                    }
                }
                
            }

            
        }
    }
}
    

    
    

//
//func addOneToRefreshCounter() -> Promise<Void> {
//    return Promise { seal in
//        guard let currentUser = Auth.auth().currentUser else {
//            seal.reject(Err("No User Profile"))
//            return
//        }
//
//        let _ = db.collection("users")
//            .document(currentUser.uid).updateData(["refreshCounter" : FieldValue.increment(Int64(1))]) { error in
//                if let error = error {
//                    seal.reject(error)
//                } else {
//                    seal.fulfill(())
//                }
//            }
//    }
//}
