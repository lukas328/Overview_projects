//
//  Enums.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 05.03.21.
//

import Foundation
import SwiftUI

// MARK: Marks the status of an event
// used in: MeMatchStartView
enum EventStatus {
    // user has no likes on his event yet (no swipe possible)
    case notLiked
    // user has a minimum of one like on his event (swiping possible)
    case liked
    // user has matched someone (further swiping not possible)
    case matched
}

// MARK: Each tabView has its own identifier
// used in: TabBarView
enum Tab {
    // tabView for the profile page/tab
    case profile
    // tabView for the marketplace page/tab
    case marketplace
    // tabView for the chat page/tab
    case chat
}

// MARK: Each StartView has its own identifier
// used in: MainStartView
enum StartPosition {
    // StartView for the first picture of the app
    case splash
    // StartView for the onboarding slides
    case onboarding
    // StartView for the login and register view
    case registerLogin
    // StartView for the profileCreation of the user
    case profileCreation
}

// MARK: The Distance of each location is shown in words
// used in: YouEventView
enum Distance: String {
    // here distance is 10 km
    case here
    // near distance is 30 km
    case near
    // far distance is over 30 km until set range
    case far
}

// MARK: The Category of each event which is possible to choose form
// used in: EventCreationView
enum Category: String, Equatable, CaseIterable {
    // event for meeting for a walk
    case walk = "Walk"
    // event for meeting in a cafe
    case cafe = "Café"
    // event for meeting eating together
    case food = "Food"
    // event for meeting for doing sports together
    case sport = "Sport"
    // event for meeting in a bar
    case bar = "Bar"
    // event for meeting for everything else which is not listed above
    case other = "Other"
    
    var localizedName: LocalizedStringKey { LocalizedStringKey(rawValue) }
}

// MARK: The Duration of each event which is possible to choose form
// used in: EventCreationView
enum Duration: String, Equatable, CaseIterable {
    // very short event with duration of 30 minutes
    case veryShort = "30"
    // short event with duration of 45 minutes
    case short = "45"
    // medium event with duration of 60 minutes
    case medium = "1h"
    // normal event with duration of 90 minutes
    case normal = "1,5h"
    // long event with duration of 120 minutes
    case long = "2h"
    // very long event with duration of more than 120 minutes
    case veryLong = "2+"
    
    var localizedName: LocalizedStringKey { LocalizedStringKey(rawValue) }
}

// MARK: The Covid19 preference for the event
// used in: EventCreationView
enum CovidPreference: String, Equatable, CaseIterable {
    // outdoor with a mask
    case outdoorMask = "Outdoors with face covering"
    // outdoor without a mask
    case outdoor = "Outdoors without face covering"
    // indoor and outdoor with mask
    case inOutMask = "In- and Outdoors with face covering"
    // user will adapt to other preferences
    case adapt = "I adapt to your preferences"
    
    var localizedName: LocalizedStringKey { LocalizedStringKey(rawValue) }
}
