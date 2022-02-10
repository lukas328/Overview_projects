//
//  FirestoreManagerMatches.swift
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

class FirestoreManagerMatches {
    
    private var db: Firestore
    
    
    init() {
        db = Firestore.firestore()
    }
    
    // MARK: - Functions executed by Match
    
    func addMatchToCurrentUser(userModel: UserModel, eventModel: EventModel, chatId: String) -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            let matchModel = MatchModel(userId: currentUser.uid, chatId: chatId, eventId: eventModel.eventId, matchedUserId: userModel.userId, unReadMessage: false)
            do {
                let _ = try db.collection("users")
                    .document(currentUser.uid)
                    .collection("matches").document(chatId).setData(from: matchModel) { error in
                        if let error = error {
                            seal.reject(error)
                        }
                        
                    }
            } catch { let error = error
                seal.reject(error)
            }
            seal.fulfill(())
        }
        
    }
    
    func addMatchToMatchedUser(userModel: UserModel,eventModel: EventModel, chatId: String) -> Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }

            let matchModel = MatchModel(userId: currentUser.uid, chatId: chatId, eventId: eventModel.eventId, matchedUserId: currentUser.uid, unReadMessage: false)
            do {
                let _ = try db.collection("users")
                    .document(userModel.userId)
                    .collection("matches").document(chatId).setData(from: matchModel) { error in
                        if let error = error {
                            seal.reject(error)
                        }
                    }
            } catch { let error = error
                seal.reject(error)
            }
            seal.fulfill(())
        }
    }
    
    func createChatRoom(currentUser: UserModel, userModel: UserModel, eventModel: EventModel, chatId: String) -> Promise<Void> {
        return Promise { seal in

                        
            let chatModel = ChatModel(chatId: chatId, eventCreatorId: currentUser.userId, matchedUserId: userModel.userId, eventId: eventModel.eventId, messages: [MessageModel(userId: "NoUser", timeStamp: Timestamp(date: Date()), messageText: "Hey, you have a match! \(currentUser.name) choosed as covid-preference: \(eventModel.covidPreferences), talk about your event settings...")])
            do {
                let _ = try db.collection("chats")
                    .document(chatId).setData(from:chatModel) { error in
                        if let error = error {
                            seal.reject(error)
                        }
                    }
            }catch { let error = error
                seal.reject(error)
            }
            seal.fulfill(())
        }
    }
    
    func getChatWithEventId(eventId: String) ->Promise<ChatModel> {
        return Promise { seal in
            db.collection("chats").whereField("eventId", isEqualTo: eventId).getDocuments { (snapshot, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    
                    if let snapshot = snapshot {
                        let chat: [ChatModel] = snapshot.documents.compactMap { doc in
                            let chat = try? doc.data(as: ChatModel.self)
                            if let chat = chat {
                                return chat
                            }
                            return nil
                            
                        }
                        DispatchQueue.main.async {
                            if chat.count >= 0 {
                                seal.fulfill(chat[0])
                            } else {
                                seal.reject(Err("No Chat Found"))
                            }
                            
                        }
                        
                    }
                }
                
                
            }
        }
    }
    
    // MARK: -  delete a Match
    
    func deleteMatchFromCurrentUser(chatId: String) ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            let _ = db.collection("users")
                .document(currentUser.uid).collection("matches").document(chatId).delete { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    
    func deleteMatchFromMatchedUser(chatId: String, matchedUserId: String) ->Promise<Void> {
        return Promise { seal in
            let _ = db.collection("users")
                .document(matchedUserId).collection("matches").document(chatId).delete { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    func deleteChat(chatId:String) ->Promise<Void> {
        return Promise { seal in
            let _ = db.collection("chats").document(chatId).delete { error in
                if let error = error {
                    seal.reject(error)
                } else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func deleteEvent(eventId: String) ->Promise<Void> {
        return Promise { seal in
            let _ = db.collection("events").document(eventId).delete { error in
                if let error = error {
                    seal.reject(error)
                } else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func deleteAllLikedUserFromEvent(eventId: String) ->Promise<Void> {
        return Promise { seal in
            let _ = db.collection("events").document(eventId).collection("likedUser").document("likedUser").delete { error in
                if let error = error {
                    seal.reject(error)
                } else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func setLikedUserAndMatchedUserToFalse(eventId: String) ->Promise<Void> {
        return Promise { seal in
            let _ = db.collection("events").document(eventId).updateData(["likedUser": false, "eventMatched": false]) { error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
    }

    
    
    
    
    
    
    

    func deleteLikedUser(eventModel: EventModel, userModel: UserModel) -> Promise<Void> {
        return Promise { seal in
            let _ = db.collection("events")
                .document(eventModel.eventId)
                .collection("likedUser")
                .document("likedUser").updateData(["likedUser" : FieldValue.arrayRemove([userModel.userId])]) { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }

        }
    }
}
    
