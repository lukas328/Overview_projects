
//
//FirestoreManagerEvent.swift
//Meet Me
//
//Created by Lukas Dech on 14.02.21.
//
//
//Funktions Beschreibung
//
//- saveEvent()
// saved ein Event Model zur collection User
// wird aufgerufen in --> EventCreationViewModel

//- getMeEvents()
// ladet alle Events des aktuellen Nutzers herunter und speichert alle im Array meEvents vom Typ EventModelObject
// wird aufgerufen in --> YouEventViewModel

//-getEvents
// return das Array meEvents welches von getMeEvent() objekte bekommt
// wird aufgerufen in --> YouEventViewModel

import Foundation
import FirebaseFirestore
import Firebase
import FirebaseFirestoreSwift
import FirebaseAuth
import PromiseKit
import MapKit
import GeoFire

class FirestoreManagerEventTest {
    
    private var db: Firestore
    
    
    init() {
        db = Firestore.firestore()
        
    }
    
    private var locationManager: LocationManager = LocationManager()
    var region = MKCoordinateRegion.defaultRegion
    // MARK: - Functions to Save events to Firebase
    
    
    func saveEvent(eventModel: EventModel, eventId: String) -> Promise<Void> {
        return Promise { seal in
            do {
                try db.collection("events").document(eventId).setData(from: eventModel)
                seal.fulfill(())
            }
            catch let error{
                seal.reject(error)
            }
        }
    }
    
    // MARK: - Functions to update events
    
    func createLikedUserArray(eventId: String) -> Promise<Void>{
        return Promise { seal in
            
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
            
            let likedUser = LikedUser(userId: currentUser.uid, likedUser: ["Im a feature not a bug"])
            let _ =  try db.collection("events")
                    .document(eventId)
                    .collection("likedUser")
                    .document("likedUser").setData(from: likedUser) { error in
                        if let error = error {
                            seal.reject(error)
                        }else {
                            seal.fulfill(())
                        }
                    }
                
                
                
            }
        }
    
    
    func addLikeToEventArray(eventId: String) -> Promise<Void>{
        return Promise { seal in
            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }
            
            do {
                let _ =  db.collection("events")
                    .document(eventId)
                    .collection("likedUser")
                    .document("likedUser")
                    .updateData(["likedUser" : FieldValue.arrayUnion([currentUser.uid])]){ error in
                        if let error = error {
                            seal.reject(error)
                        } else {
                            seal.fulfill(())
                        }
                    }
                
                
            }
        }
    }
    
    func setLikedUserToTrue(eventId: String) -> Promise<Void> {
        return Promise{ seal in
            let _ = db.collection("events").document(eventId).updateData(["likedUser": true]) { error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func setEventMatchedToTrue(eventId: String) -> Promise<Void> {
        return Promise{ seal in
            let _ = db.collection("events").document(eventId).updateData(["eventMatched": true]) { error in
                if let error = error {
                    seal.reject(error)
                }else {
                    seal.fulfill(())
                }
            }
        }
    }
    
    func setUpdatedEvent(eventModel: EventModel) -> Promise<Void> {
        return Promise { seal in
            let _ = db.collection("events")
                .document(eventModel.eventId)
                .updateData(["category": eventModel.category,
                             "date": eventModel.date,
                            "endTime": eventModel.endTime,
                            "startTime": eventModel.startTime]){ error in
                    if let error = error {
                        seal.reject(error)
                    }else {
                        seal.fulfill(())
                    }
                }
        }
    }
    
    
    // MARK: - Functions to get events
    func firebaseGetYouEventsTest(likedEvents : [String]) -> Promise<[EventModel]> {
        return Promise { seal in


            guard let currentUser = Auth.auth().currentUser else {
                throw Err("No User Profile")
            }

            db.collection("events").whereField("eventMatched", isEqualTo: false)
                .getDocuments{(snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {

                        if let snapshot = snapshot {
                            var event: [EventModel]? = snapshot.documents.compactMap { doc in
                                var event = try? doc.data(as: EventModel.self)
                                event?.eventId = doc.documentID
                                if let event = event {
                                    if event.userId != currentUser.uid && event.eventMatched == false {
                                        return event
                                    }
                                }
                                return nil

                            }
                            if event != nil {
                                for (index, eventModel) in event!.enumerated().reversed() {
                                    if likedEvents.contains(eventModel.eventId) {
                                        event!.remove(at: index)
                                    }
                                }
                                DispatchQueue.main.async {
                                    seal.fulfill(event!)
                                }
                            } else {
                                let error = Err("No Events in GetYouEvents")
                                DispatchQueue.main.async {
                                    seal.reject(error)
                                }
                            }
                        }



                    }
                }


        }
    }
    
    func getAllLikedUserDocument(eventId: String) -> Promise<[String]> {
        return Promise { seal in

            
            db.collection("events")
                .document(eventId)
                .collection("likedUser")
                .document("likedUser")
                .getDocument { (snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        if let snapshot = snapshot {
                            let likedUser = try? snapshot.data(as: LikedUser.self)
                            if likedUser != nil {
                                
                                DispatchQueue.main.async {
                                    if likedUser?.likedUser.count != 0 {
                                        seal.fulfill(likedUser!.likedUser)
                                    } else {
                                        seal.reject(Err("Liked document is empty"))
                                    }
                                }
                            }
                            
                        }
                        
                        
                    }
                }
        }
    }
    
    
    func getAllLikedUserModels(likedUser: [String]) -> Promise<[UserModel]> {
        return Promise { seal in
            db.collection("users")
                .whereField("userId", in: likedUser)
                .getDocuments {(snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    } else {
                        
                        if let snapshot = snapshot {
                            let userModel: [UserModel] = snapshot.documents.compactMap { doc in
                                let userModel = try? doc.data(as: UserModel.self)
                                if let userModel = userModel {
                                    return userModel
                                }
                                return nil
                                
                            }
                            DispatchQueue.main.async {
                                seal.fulfill(userModel)
                            }
                            
                        }
                        
                    }
                }
            
            
        }
    }
    
    func queryColletion(center : CLLocationCoordinate2D, user: UserModel) -> Promise<[Query]> {
        return Promise { seal in
            //let center = CLLocationCoordinate2D(latitude: 53.5466399, longitude: 9.93079)
            //let radiusInKilometers: Double = 50
            
            let queryBounds = GFUtils.queryBounds(forLocation: center,
                                                  withRadius: user.radiusInKilometer * 1000)
            let queries = queryBounds.compactMap { (any) -> Query? in
                guard let bound = any as? GFGeoQueryBounds else { return nil }
                return db.collection("events")
                    .order(by: "hash")
                    .start(at: [bound.startValue])
                    .end(at: [bound.endValue])
            }
            seal.fulfill(queries)
            
        }
    }
    
    func querysInEvent(likedEvents: [String], queries: [Query], center : CLLocationCoordinate2D, user: UserModel, shuffle: Bool, reportedEvents : [String]) ->Promise<[EventModel]> {
        return Promise { seal in

            let userPoint = CLLocation(latitude: center.latitude, longitude: center.longitude)
            
            
            guard let currentUser = Auth.auth().currentUser else {
                seal.reject(Err("No User Profile"))
                return
            }
            
            for query in queries {
                query.getDocuments {(snapshot, error) in
                    if let error = error {
                        seal.reject(error)
                    }else {
                        
                        if let snapshot = snapshot {
                            var event: [EventModel]? = snapshot.documents.compactMap { doc in
                                var event = try? doc.data(as: EventModel.self)
                                event?.eventId = doc.documentID
                                if var event = event {
                                    if event.userId != currentUser.uid && event.eventMatched == false {
                                       // print("DEBUG geschlecht abfrage: event.serchingFor: \(event.searchingFor)  == user.gender: \(user.gender) && event.genderFromCreator: \(event.genderFromCreator) ==  user.searchingFor:  \(user.searchingFor)")
                                        
                                        if (event.searchingFor == user.gender && event.genderFromCreator == user.searchingFor) || user.searchingFor == "All"{
                                            let eventPoint = CLLocation(latitude: event.latitude, longitude: event.longitude)
                                            event.distance = GFUtils.distance(from: userPoint, to: eventPoint) / 1000
                                            //print("EventDistance: \(event.distance)")
                                            //return EventModelObject(eventModel: event, position: .constant(CGSize.zero))
                                            return event
                                        }
                                    }
                                }
                                return nil
                            }
                            if event != nil {
                                
                                var eventsSorted: [EventModel] = []
                                if shuffle {
                                    event!.shuffle()
                                }
                                print(event!.count)
                                if event!.count > 0 {
                                    for (index, eventModel) in event!.enumerated().reversed() {
                                        if reportedEvents.contains(eventModel.eventId) == false {
                                            if likedEvents.contains(eventModel.eventId) == false {
                                                //event!.remove(at: index)
                                                eventsSorted.append(event![index])
                                            }
                                        }
                                        if eventsSorted.count >= 15 {
                                            break
                                        }
                                    }
                                    print(event!.count)
                                    DispatchQueue.main.async {
                                        print(event!.count)
                                        print(eventsSorted)
                                        //seal.fulfill(Array(eventsSorted.prefix(10)))
                                        seal.fulfill(eventsSorted)
                                    }
                                }
                            }else {
                                
                                let error = Err("No Events in GetYouEvents")
                                DispatchQueue.main.async {
                                    seal.reject(error)
                                    
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}














