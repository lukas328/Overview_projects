//
//  OnboardingController.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 14.03.21.
//

import SwiftUI

struct OnboardingController: View {
    
    @Binding var showOnboarding: StartPosition
    
    @State var goToNextView: Bool = false
    @State var goToLastView: Bool = false
    
    var sliderArray: [InformationCardModel] = [InformationCardModel(), InformationCardModel(headerText: "Do this with DAISY instead!", highlight: true, image: "walking", sfSymbol: "person.2.fill", buttonText: "OK!"), InformationCardModel(headerText: "Meeting new people", highlight: true, footerText: "easy as going shopping", image: "shopping", sfSymbol: "cart.fill", buttonText: "OK!"), InformationCardModel(headerText: "Meeting new people", highlight: true, footerText: "normal as walking the dog", image: "dog", sfSymbol: "hare.fill", buttonText: "OK!"), InformationCardModel(headerText: "Meeting new people", highlight: true, footerText: "safely as chilling at home", image: "chilling", sfSymbol: "lock.shield.fill", buttonText: "GO!")]
    
    var body: some View {
        
        InformationCard(goToNextView: $goToNextView, goToLastView: $goToLastView, sliderArray: sliderArray, onBoardingUsage: true)
        .onChange(of: goToNextView, perform: { value in
            showOnboarding = .registerLogin
        })
        .onChange(of: goToLastView, perform: { value in
            showOnboarding = .splash
        })
    }
}

struct OnboardingController_Previews: PreviewProvider {
    static var previews: some View {
        OnboardingController(showOnboarding: .constant(.onboarding))
    }
}
