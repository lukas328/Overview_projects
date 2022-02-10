//
//  FirestoreManagerUserTest.swift
//  Meet Me
//
//  Created by Lukas Dech on 21.02.21.
//

import Foundation
import Firebase
import FirebaseFirestoreSwift
import FirebaseFirestore
import FirebaseAuth
import PromiseKit

class FirestoreManagerUserTest {
    
    private var db: Firestore
    
    
    init() {
        db = Firestore.firestore()
        
    }
    
    
    
    
    // MARK: - Functions to Save userModel to Firebase
    func saveUser(userModel: UserModel) -> Promise<Void>{
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser?.uid else {
                throw Err("no current User")
            }
            do {
                try db.collection("users").document(currentUser).setData(from: userModel)
                seal.fulfill(())
            } catch let error {
                seal.reject(error)
            }
        }
    }
    
    func createLikedEventsArray() -> Promise<Void>{
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            do {
                let _ =  db.collection("users")
                    .document(currentUser.uid)
                    .collection("likedEvent")
                    .document("likedEvent").setData(["likedEvent": ["Im not a Bug Im a feature"]])
                
                seal.fulfill(())
                
            }
        }
    }
    
    func createReportedEventsArray() -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            do {
                let _ =  db.collection("users")
                    .document(currentUser.uid)
                    .collection("reportedEvents")
                    .document("reportedEvents").setData(["reportedEvents": ["Im not a Bug Im a feature"]])
                
                seal.fulfill(())
                
            }
        }
    }
    
    func addLikeToEventArray(eventId: String) -> Promise<Void>{
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            do {
                let _ =  db.collection("users")
                    .document(currentUser.uid)
                    .collection("likedEvent")
                    .document("likedEvent")
                    .updateData(["likedEvent" : FieldValue.arrayUnion([eventId])])
                seal.fulfill(())
                
            }
        }
    }
    
    func addReportToReportArray(eventId: String) -> Promise<Void>{
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            do {
                let _ =  db.collection("users")
                    .document(currentUser.uid)
                    .collection("reportedEvents")
                    .document("reportedEvents")
                    .updateData(["reportedEvents" : FieldValue.arrayUnion([eventId])])
                seal.fulfill(())
                
            }
        }
    }
    
    func deletePushNotificationTokenFromUser() ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            let _ = db.collection("tokens").document(currentUser.uid).updateData(["token" : ""]) { error in
                if let error = error {
                    seal.reject(error)
                } else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    
    func addOneToRefreshCounter() -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            let _ = db.collection("users")
                .document(currentUser.uid).updateData(["refreshCounter" : FieldValue.increment(Int64(1))]) { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    func addOneToReportCounter(userId: String) -> Promise<Void> {
        return Promise { seal in
            
            let _ = db.collection("users")
                .document(userId).updateData(["reports" : FieldValue.increment(Int64(1))]) { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    
    // MARK: - Functions to Update current User
    func setRadius(radius: Double) -> Promise<Void>{
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
            let _ = db.collection("users")
                .document(currentUser.uid).updateData(["radiusInKilometer": radius]) { error in
                    if let error = error {
                        seal.reject(error)
                    }else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    
    func setSearchingForUserProfile(searchingFor: String) -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {return}
            
            let _ =  db.collection("users").document(currentUser.uid).updateData(["searchingFor" : searchingFor]){ error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
        
    }
    func setSearchingForEvents(searchingFor: String) ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
            db.collection("events").whereField("userId", isEqualTo: currentUser.uid).getDocuments{ (snapshot, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    let ids: [String]! = snapshot?.documents.compactMap { doc in
                        return doc.documentID
                    }
                    for id in ids {
                        self.db.collection("events").document(id).updateData(["searchingFor" : searchingFor]){ error in
                            if let error = error {
                                seal.reject(error)
                            }
                        }
                    }
                    seal.fulfill(())
                    
                }
            }
            
        }
        
    }
    
    func setLastLogin() ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No Current User"))
                return
            }
            
            let _ =  db.collection("users").document(currentUser.uid).updateData(["lastLogin" : getShortDate()]){ error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func setRefreshCounterToZero() ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No Current User"))
                return
            }
            let _ =  db.collection("users").document(currentUser.uid).updateData(["refreshCounter" : 0 ]){ error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
            
        }
    }
    
    
    
    
    
    // MARK: - Functions to get User Profiles
    
    func getAllLikedEvents() -> Promise<[String]> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
            
            db.collection("users")
                .document(currentUser.uid)
                .collection("likedEvent")
                .document("likedEvent")
                .getDocument { (snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        if let snapshot = snapshot {
                            let likedEvent = try? snapshot.data(as: LikedEvent.self)
                            if likedEvent != nil {
                                
                                DispatchQueue.main.async {
                                    seal.fulfill(likedEvent!.likedEvent)
                                    
                                }
                            }
                            
                        }
                        
                        
                    }
                }
        }
    }
    
    func getAllReportedEvents() -> Promise<[String]> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
            
            db.collection("users")
                .document(currentUser.uid)
                .collection("reportedEvents")
                .document("reportedEvents")
                .getDocument { (snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        if let snapshot = snapshot {
                            let reportedEvents = try? snapshot.data(as: ReportedEvents.self)
                            if reportedEvents  != nil {
                                
                                DispatchQueue.main.async {
                                    
                                    seal.fulfill(reportedEvents!.reportedEvents)
                                }
                            }
                            
                        }
                        
                        
                    }
                }
        }
    }
    
    
    func getCurrentUser() -> Promise<UserModel> {
        return Promise { seal in
            
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            db.collection("users").document(currentUser.uid).getDocument { snapshot, error in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let userModel = try? snapshot.data(as: UserModel.self)
                        DispatchQueue.main.async {
                            if userModel != nil {
                                seal.fulfill(userModel!)
                            }else {
                                let err = Err("userModel")
                                seal.reject(err)
                            }
                        }
                        
                        
                    }
                }
                
            }
            
            
        }
    }
    
    
    func getUserWhichCreatedEvent(eventModel: EventModel) -> Promise<UserModel> {
        return Promise { seal in
            db.collection("users").document(eventModel.userId).getDocument { snapshot, error in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let userModel = try? snapshot.data(as: UserModel.self)
                        DispatchQueue.main.async {
                            if userModel != nil{
                                seal.fulfill(userModel!)
                            } else {
                                let error = Err("Cant get UserProfil from Creator")
                                seal.reject(error)
                            }
                        }
                        
                    }
                }
            }
        }
    }
    
    func getUserWithUserId(userId: String) -> Promise<UserModel> {
        return Promise { seal in
            db.collection("users").document(userId).getDocument { snapshot, error in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let userModel = try? snapshot.data(as: UserModel.self)
                        DispatchQueue.main.async {
                            if userModel != nil{
                                seal.fulfill(userModel!)
                            } else {
                                let error = Err("Cant get UserProfil from Creator")
                                seal.reject(error)
                            }
                        }
                        
                    }
                }
            }
        }
    }
}







