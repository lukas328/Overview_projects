//
//  FirestoreManagerChat.swift
//  Meet Me
//
//  Created by Lukas Dech on 27.02.21.
//

import Foundation
import Firebase
import FirebaseFirestoreSwift
import FirebaseFirestore
import FirebaseAuth
import PromiseKit

class FirestoreManagerChat: ObservableObject  {
    
    private var db: Firestore
    
    init() {
        db = Firestore.firestore()
    }

    // MARK: - Functions to Download Matches

    func getAllMatchDocumentsCurrentUser() ->Promise<[MatchModel]> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            
            db.collection("users")
                .document(currentUser.uid)
                .collection("matches")
                .getDocuments { (snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        if let snapshot = snapshot {
                            let matchDocument: [MatchModel] = snapshot.documents.compactMap { doc in
                                let matchDocument = try? doc.data(as: MatchModel.self)
                                if let matchDocument = matchDocument {
                                    return matchDocument
                                }
                                return nil
                            }
                            DispatchQueue.main.async {
                                seal.fulfill(matchDocument)
                                
                            }
                        }
                    }
                }
            
        }
        
    }
    
    
    func getUserWithUserId(userId: String) ->Promise<UserModel> {
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
    


    
    func getEventWithEventId(eventId: String) ->Promise<EventModel> {
        return Promise { seal in
            db.collection("events").document(eventId).getDocument { snapshot, error in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let eventModel = try? snapshot.data(as: EventModel.self)
                        DispatchQueue.main.async {
                            if eventModel != nil{
                                seal.fulfill(eventModel!)
                            } else {
                                let error = Err("Cant get Event from Creator")
                                seal.reject(error)
                            }
                        }
                        
                    }
                }
            }
        }
    }
    
    
    // MARK: - Functions to Download Chat and an Upload Messages
    
    

    
    
    func uploadMessage(messageText: String, chatId: String) -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            if messageText != "" {
                let timeStamp: Timestamp = Timestamp(date: Date())
                let messageModel = MessageModel(userId: currentUser.uid,timeStamp: timeStamp, messageText: messageText)
                let _ = db.collection("chats")
                    .document(chatId)
                    .updateData(["messages" : FieldValue.arrayUnion([messageModel.dictionary])]) { error in
                        if let error = error {
                            seal.reject(error)
                        } else {
                            seal.fulfill(())
                        }
                    }
            }else {
                seal.reject(Err("message was empty"))
            }
        }
    }
    
    func setUnReadMessageToTrue(chatId: String, userIdReciever: String) -> Promise<Void> {
        return Promise { seal in
            
            let _ = db.collection("users")
                .document(userIdReciever)
                .collection("matches")
                .document(chatId)
                .updateData(["unReadMessage" : true]) { error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func setUnReadMessageToFalse(chatId: String) -> Promise<Void> {
        return Promise { seal in
            
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            
            let _ = db.collection("users")
                .document(currentUser.uid)
                .collection("matches")
                .document(chatId)
                .updateData(["unReadMessage" : false]) { error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
    }
    
}


    

