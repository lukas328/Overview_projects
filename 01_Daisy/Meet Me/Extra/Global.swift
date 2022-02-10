//
//  GlobalFunctions.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 14.02.21.
//

import Foundation
import Firebase
import FirebaseFirestoreSwift
import SwiftUI

// MARK: Get information from screen etc - NOT USED ANYMORE, USE GEOMETRY READER INSTEAD
// get the size of the screen (defined here that it can be used in every View)
//let screen = UIScreen.main.bounds


// MARK: - Functions

// MARK: Function to get the age from a date object
func dateToAge(date: Date) -> Int {
    let lifeTime = date.timeIntervalSinceNow
    let years = -lifeTime / 60 / 60 / 24 / 365
    return Int(years)
}

// MARK: Create a sample date
func createSampleDate() -> Date {
    
    var dateComponents = DateComponents()
    dateComponents.year = 1998
    dateComponents.month = 9
    dateComponents.day = 6
    
    // since the components above (like year 1980) are for Gregorian
    let userCalendar = Calendar(identifier: .gregorian)
    
    let someDateTime = userCalendar.date(from: dateComponents)!
    return someDateTime
}

// MARK: Create a UUID as a string
var createdId = UUID().uuidString

func getShortDate() -> String {
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "dd/MM/yyyy"
    return dateFormatter.string(from: Date())
}

// MARK: - Colors

// MARK: Gradient which is the secondary color (blueish) and marks cancel and abort actions
let secondaryGradient = LinearGradient(
    gradient: Gradient(stops: [
                        .init(color: Color(#colorLiteral(red: 0.0313725508749485, green: 0.45490196347236633, blue: 0.5490196347236633, alpha: 1)), location: 0),
                        .init(color: Color(#colorLiteral(red: 0.0313725471496582, green: 0.4549018144607544, blue: 0.5490196347236633, alpha: 0.6299999952316284)), location: 1)]),
    startPoint: UnitPoint(x: 0.9999999999999999, y: 7.105427357601002e-15),
    endPoint: UnitPoint(x: -2.220446049250313e-16, y: -1.7763568394002505e-15)
)


// MARK: - Stockdata

// MARK: stock user to use as default
let stockUser: UserModel = UserModel(userId: "007", name: "Stocky One", birthdayDate: createSampleDate(), gender: "Male", searchingFor: "Male", userPhotos: [1: stockUrlString], userPhotosId: [0: ""], radiusInKilometer: 1000, refreshCounter: 0, userStatus: "normal",reports: 0, loginToken: "", lastLogin: getShortDate())
let stockUser2: UserModel = UserModel(userId: "008", name: "Stockise Two", birthdayDate: createSampleDate(), gender: "Female", searchingFor: "Male", userPhotos: [0: stockUrlString], userPhotosId: [1: ""], radiusInKilometer: 1000, refreshCounter: 0, userStatus: "normal",reports: 0, loginToken: "", lastLogin: getShortDate())

// TODO: Need to be changed to a stock photo
let stockURL: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2Fphilipp%401024x.png?alt=media&token=6ce1af89-a75c-45a4-a698-ff0a9ac68f46")!
let stockUrlString: String = "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2Fphilipp%401024x.png?alt=media&token=6ce1af89-a75c-45a4-a698-ff0a9ac68f46"

let stockMale: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2Fmale-light%20Kopie.jpg?alt=media&token=cbcfc58a-4dec-43dc-ba19-66946f5d1f68")!
let stockMaleString: String = "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2Fmale-light%20Kopie.jpg?alt=media&token=cbcfc58a-4dec-43dc-ba19-66946f5d1f68"
let stockFemale: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2Ffemale-light%20Kopie.jpg?alt=media&token=68f407fc-6651-4592-a553-5ab99fed85f9")!
let stockFemaleString: String = "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2Ffemale-light%20Kopie.jpg?alt=media&token=68f407fc-6651-4592-a553-5ab99fed85f9"

//let stockUrlBar: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2FBar.jpg?alt=media&token=628505b0-d3c4-429c-9e81-14ff75479a26")!
//let stockUrlCoffee: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2FCoffee.jpg?alt=media&token=64a3caa8-7209-4cc7-8011-d3032fcfff8e")!
//let stockUrlSport: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2FSport.jpeg?alt=media&token=652b2c84-b94a-4463-ab08-b381ba8d6605")!
//let stockUrlWalk: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2FWalk.jpeg?alt=media&token=d7f07da6-f962-49b1-8289-4cbecd7f5b5c")!
//let stockUrlFood: URL = URL(string: "https://firebasestorage.googleapis.com/v0/b/meetme-1c961.appspot.com/o/StockImages%2FResaurant.jpg?alt=media&token=6c01f295-a8c5-4c62-bed6-571758ac6e47")!



// MARK: stock event for marketplace and eventObject (length: half an hour - 30*60 seconds)
let stockEvent: EventModel = EventModel(eventId: "008", userId: "007", category: "Café", date: Date(), startTime: Date(), endTime: Date() + 30 * 60, pictureURL: "", eventPhotosId: "", profilePicture: "",likedUser: false,eventMatched: false,latitude: 0.0, longitude: 0.0, hash: "", distance: 0, searchingFor: "Male", genderFromCreator: "Female", birthdayDate: Date(),covidPreferences: "")
let stockEvent2: EventModel = EventModel(eventId: "009", userId: "008", category: "Walk", date: Date(), startTime: Date(), endTime: Date() + 30 * 60, pictureURL: "", eventPhotosId: "", profilePicture: "",likedUser: false,eventMatched: false,latitude: 0.0, longitude: 0.0, hash: "", distance: 0,searchingFor: "Male", genderFromCreator: "Female", birthdayDate: Date(),covidPreferences: "")

// MARK: stock chat for chat area
let stockChat: ChatModel = ChatModel(chatId: "egal", eventCreatorId: "creatoRR", matchedUserId: "receiveRR", eventId: "egal2", messages: [MessageModel(userId: "", timeStamp: Timestamp(date: Date()), messageText: "Hey, we have a match! Speak with YOU about the event settings...")])


//PushNotifications
let notificationLikeEventTitle = "New Like!"
let notificationLikeEventMessage = "Tap on the ME Event to match."

let notificationMatchMessageTitle = "New match!"
let notificationMatchMessage = "Start setting up the event details."

//username gets written in front of message
let notificationMessageTitle = "New message!"
let notificationMessageMessage = "sent you a message."


