//
//  DeveloperViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 27.03.21.
//

import Foundation
import PromiseKit
import Firebase



class DeveloperViewModel: ObservableObject {
    private var firestoreManagerMatches: FirestoreManagerMatches = FirestoreManagerMatches()
    private var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()
    private var firestoreFotoManagerUserTest: FirestoreFotoManagerUserTest = FirestoreFotoManagerUserTest()
    private var firestoreManagerEventTest: FirestoreManagerEventTest = FirestoreManagerEventTest()
    private var firestoreFotoManagerEventTest: FirestoreFotoManagerEventTest = FirestoreFotoManagerEventTest()
    private var firestoreManagerChat: FirestoreManagerChat = FirestoreManagerChat()
    private var developerManager: DeveloperManager = DeveloperManager()
    private var db: Firestore
    
    private var Ids: [String] = []
    

    private var oldEventsWithoutMatch: [String] = []
    private var oldEventsWithMatch: [String] = []
    private var chatModels: [ChatModel] = []
    
    
    init() {
        db = Firestore.firestore()
        
    }
    
    func setShuffelCouterToZeroAllUsers() {
        firstly {
            getAllUserIDs()
        }.then { userIds in
            when(fulfilled: self.Ids.compactMap(self.setRefreshcounterTozero))
        }.done {
            print("done")
        }.catch { error in
            print(error)
        }
    }
    
    func getAllUserIDs() ->Promise<Void> {
        return Promise { seal in
            let _ = db.collection("users").getDocuments {(snapshot, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let userIDs: [String] = snapshot.documents.compactMap { doc in
                            let user = try? doc.data(as: UserModel.self)
                            if let user = user {
                                return user.userId
                            }
                            return nil
                        }
                        DispatchQueue.main.async {
                            self.Ids = userIDs
                            seal.fulfill(())
                        }
                    }
                }
            }
        }
    }
    
    func setRefreshcounterTozero(userId: String) -> Promise<Void> {
        return Promise { seal in
            let _ = db.collection("users")
                .document(userId).updateData(["refreshCounter": 0]) { error in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    
    func deleteAllOldEvents() {
        firstly {
            getAllEvents()
        }.then { events in
            self.checkEvents(events: events)
        }.then {
            when(fulfilled: self.oldEventsWithoutMatch.compactMap(self.firestoreManagerMatches.deleteEvent))
        }.then {
            when(fulfilled: self.oldEventsWithMatch.compactMap(self.getAllChatsWithEventId)).done {
                when(fulfilled: self.chatModels.compactMap(self.deleteMatches)).done {
                    print("done")
                }.catch { error in
                    print(error)
                }
            }
        }.catch { error in
            print(error)
        }
    }
        
    func deleteMatches(chat: ChatModel) ->Promise<Void> {
        return Promise { seal in
            firstly {
                when(fulfilled:
                     self.firestoreManagerMatches.deleteMatchFromMatchedUser(chatId: chat.chatId, matchedUserId: chat.eventCreatorId),
                     self.firestoreManagerMatches.deleteMatchFromMatchedUser(chatId: chat.chatId, matchedUserId: chat.matchedUserId),
                     self.firestoreManagerMatches.deleteAllLikedUserFromEvent(eventId: chat.eventId),
                     self.firestoreManagerMatches.deleteChat(chatId: chat.chatId),
                     self.firestoreManagerMatches.deleteEvent(eventId: chat.eventId))
            }.catch { error in
                seal.reject(error)
            }
        }
    }

    
    
    
    func getAllEvents() -> Promise<[EventModel]> {
        return Promise { seal in
            let _ = db.collection("events").getDocuments {(snapshot, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let events: [EventModel] = snapshot.documents.compactMap { doc in
                            let event = try? doc.data(as: EventModel.self)
                            if let event = event {
                                return event
                            }
                            return nil
                        }
                        DispatchQueue.main.async {

                            seal.fulfill(events)
                        }
                    }
                }
            }
        }
    }
    
    func getAllChatsWithEventId(eventId: String) -> Promise<Void> {
        return Promise { seal in
            let _ = db.collection("chats").whereField("eventId", isEqualTo: eventId).getDocuments {(snapshot, error) in
                if let error = error {
                    seal.reject(error)
                } else {
                    if let snapshot = snapshot {
                        let chats: [ChatModel] = snapshot.documents.compactMap { doc in
                            let chat = try? doc.data(as: ChatModel.self)
                            if let chat = chat {
                                return chat
                            }
                            return nil
                        }
                        DispatchQueue.main.async {
                            for chat in chats {
                                self.chatModels.append(chat)
                            }
                                
                            seal.fulfill(())
                        }
                    }
                }
            }
        }
    }
    
    
    func checkEvents(events: [EventModel]) ->Promise<Void> {
        return Promise { seal in
            let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
            for event in events {
                if event.date < yesterday {
                    if event.eventMatched {
                        oldEventsWithMatch.append(event.eventId)
                    }else {
                        oldEventsWithoutMatch.append(event.eventId)
                    }
                }
            }
            seal.fulfill(())
        }
    }
    
    func setMaintenanceModeToTrue() {
        firstly {
            developerManager.setMaintenanceMode(status: true)
        }.done {
            print("setMaintenanceModeToTrue, done")
        }.catch { error in
            print(error)
        }
    }
    
    func setMaintenanceModeToFalse() {
        firstly {
            developerManager.setMaintenanceMode(status: false)
        }.done {
            print("setMaintenanceModeToFalse, done")
        }.catch { error in
            print(error)
        }
    }
    
    

    
}
