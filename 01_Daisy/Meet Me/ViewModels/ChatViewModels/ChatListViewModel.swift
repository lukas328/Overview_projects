//
//  ChatListViewModel.swift
//  Meet Me
//
//  Created by Lukas Dech on 28.02.21.
//


//Alle MatchInformationModels
import Foundation
import PromiseKit
import Firebase


class ChatListViewModel: ObservableObject {
    
    private var firestoreManagerChat: FirestoreManagerChat = FirestoreManagerChat()
    private var firestoreManagerMatches: FirestoreManagerMatches = FirestoreManagerMatches()
    private var firestoreManagerEventTest: FirestoreManagerEventTest = FirestoreManagerEventTest()
    private var firestroeManagerFotoEventTest: FirestoreFotoManagerEventTest = FirestoreFotoManagerEventTest()
    private var firestoreManagerUserTest: FirestoreManagerUserTest = FirestoreManagerUserTest()
    private var matchDoc : [MatchModel] = []
    @Published var matches : [AllMatchInformationModel] = []
    @Published var messageIfNoMatches = ""
    private var db: Firestore
    private var currentUserId = ""
    
    
    init() {
        db = Firestore.firestore()
        

    }
    
    func checkIfUserIsCreatorFromEvent(creatorId: String) -> Bool {
        guard let currentUser = Auth.auth().currentUser else {
            return false
        }
        if currentUser.uid == creatorId {
            return true
        }else {
            return false
        }
    }
    
    func activateListner() ->Promise<Void> {
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                return
            }
            
            db.collection("users")
                .document(currentUser.uid)
                .collection("matches")
                .addSnapshotListener { (snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        if let _ = snapshot {
                            self.getMatches()
                            seal.fulfill(())
                        }
                        
                    }
                }
        }
    }
    
        

    
    
    func getMatches() {
        firstly {
            self.firestoreManagerChat.getAllMatchDocumentsCurrentUser()
        }.map { matchDocs in
            self.matchDoc = matchDocs
        }.then {
            when(fulfilled: self.matchDoc.compactMap(self.getAllMatchInformation)).done { result in
                self.matches = result
            }.done{
                if self.matches.count != 0 {
                    self.matches = self.matches.sorted{
                        ($0.event.date,$0.event.startTime ) <
                            ($1.event.date,$1.event.startTime )
                       
                    }
                }
                else {
                    self.messageIfNoMatches = "Drag Events at the Meet ME Market and match!"
                }
            }
        }.catch { error in
            print("DEBUG: error in getMatchesChain error: \(error)")
            self.messageIfNoMatches = "Drag Events and Match!"
            print(error.localizedDescription)
        }
    }
    
    
    func getAllMatchInformation(doc: MatchModel) -> Promise<AllMatchInformationModel> {
        return Promise { seal in
            firstly{
                when(fulfilled: self.firestoreManagerChat.getEventWithEventId(eventId: doc.eventId),
                     self.firestoreManagerChat.getUserWithUserId(userId: doc.matchedUserId))
            }.done{ event, user in
                let matchInformation = AllMatchInformationModel(chatId: doc.chatId, unReadMessage: doc.unReadMessage, user: user, event: event)
                seal.fulfill(matchInformation)
            }.catch { error in
                seal.reject(error)
            }
        }
        
    }
    
    
    func reportUser(match: AllMatchInformationModel) {
        guard let currentUser = Auth.auth().currentUser else {
            return
        }
        
        if currentUser.uid == match.event.userId {
            deleteMatchAndEventCompletely(match: match, index: 0)
        } else {
            deleteMatchAndBackToPool(match: match, index: 0)
        }
        firestoreManagerUserTest.addOneToReportCounter(userId: match.user.userId).catch { error in
            print(error.localizedDescription)
        }
        
    }
    
    func deleteMatchAndEventCompletely(match: AllMatchInformationModel, index: Int) {
        guard let currentUser = Auth.auth().currentUser else {
            return
        }
        
        if currentUser.uid == match.event.userId {
            firstly {
                when(fulfilled:
                        self.firestoreManagerMatches.deleteMatchFromCurrentUser(chatId: match.chatId),
                     self.firestoreManagerMatches.deleteMatchFromMatchedUser(chatId: match.chatId, matchedUserId: match.user.userId),
                     self.firestoreManagerMatches.deleteChat(chatId: match.chatId),
                     self.firestoreManagerMatches.deleteAllLikedUserFromEvent(eventId: match.event.eventId),
                     self.firestoreManagerMatches.deleteEvent(eventId: match.event.eventId),
                     self.firestroeManagerFotoEventTest.deleteImageFromStorage(storageId: match.event.eventPhotosId))
                
            }.done {
                //self.matches.remove(at: index)
            }.catch { error in
                print("DEGUB: error in deleteMatch complete, error: \(error)")
                print("DEGUB: error localized: \(error.localizedDescription)")
            }
        }else {
            deleteMatchAndBackToPool(match: match, index: index)
        }
        
    }
    
    func deleteMatchAndBackToPool(match: AllMatchInformationModel, index: Int) {
        firstly {
            when(fulfilled:
                 self.firestoreManagerMatches.deleteMatchFromCurrentUser(chatId: match.chatId),
                 self.firestoreManagerMatches.deleteMatchFromMatchedUser(chatId: match.chatId, matchedUserId: match.user.userId),
                 self.firestoreManagerMatches.deleteChat(chatId: match.chatId),
                 self.firestoreManagerMatches.deleteAllLikedUserFromEvent(eventId: match.event.eventId),
                 self.firestoreManagerEventTest.createLikedUserArray(eventId: match.event.eventId),
                 self.firestoreManagerMatches.setLikedUserAndMatchedUserToFalse(eventId: match.event.eventId))
        }.done {
            //self.matches.remove(at: index)

        }.catch { error in
            print("DEGUB: error in deleteMatch complete, error: \(error)")
            print("DEGUB: error localized: \(error.localizedDescription)")
        }
    }


}


