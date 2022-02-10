//
//  OnboardingSlideView.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 14.03.21.
//

import SwiftUI

struct InformationCard: View {
    
    @Binding var goToNextView: Bool
    @Binding var goToLastView: Bool
    
    @State var index: Int = 0
    var sliderArray: [InformationCardModel]
    
    var onBoardingUsage: Bool = false
    let notchPhone: Bool = UIApplication.shared.windows[0].safeAreaInsets.bottom > 0 ? true : false
    
    var body: some View {
        
        GeometryReader { bounds in
            VStack {
                
                Spacer()
                
                
                ZStack {
                    VStack(alignment: .leading) {
                        
                        // MARK: Top of the card
                        if sliderArray[index].highlight {
                            
                            // highlighted heading
                            Text(sliderArray[index].headerText)
                                .font(.largeTitle)
                                .gradientForeground(gradient: secondaryGradient)
                                .padding(.trailing, 55)
                        } else {
                            
                            // normal heading
                            Text(sliderArray[index].headerText)
                                .font(.largeTitle)
                                .padding(.trailing, 55)
                        }
                        
                        // MARK: Subheading
                        if sliderArray[index].footerText != "" {
                            Text(sliderArray[index].footerText)
                                .font(.title)
                                .padding(.top, 5)
                        }
                        
                        Spacer()
                        
                        // MARK: Main info/describing text
                        if sliderArray[index].subtext != "" {
                            Text(sliderArray[index].subtext)
                                .font(.body)
                                .padding(.top, 5)
                        }
                        
                        // MARK: Image
                        Image(sliderArray[index].image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .padding(.top, 20)
                        
                        Spacer()
                        
                        // MARK: Bottom dots to show how many cards are left
                        if sliderArray.count > 1 {
                            HStack(spacing: 5.0) {
                                ForEach(sliderArray.indices, id: \.self) { sliderIndex in
                                    Capsule()
                                        .frame(width: sliderIndex == index ? 24 : 13, height: 13)
                                        .foregroundColor(sliderIndex == index ? Color("BackgroundSecondary") : .gray)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .center)
                        }
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    
                    .frame(width: (bounds.size.width - 48), height: (bounds.size.width - 48) * 1.33 + 40)
                    .background(
                        Image("me-event-background")
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .scaleEffect(1.1)
                    )
                    .modifier(offWhiteShadow(cornerRadius: 26))
                    .padding(.horizontal, 48)
                    
                    
                    iconCircle
                        .offset(x: (bounds.size.width - 48) / 2 - 30, y: -((bounds.size.width - 48) * 1.33) / 2 + 10)
                    
                    // MARK: Tapping area for changing pictures
                    HStack {
                        // left tap
                        Color.black.opacity(0.001)
                            .onTapGesture {
                                goBackward()
                            }
                        
                        // right tap
                        Color.black.opacity(0.001)
                            .onTapGesture {
                                goForward()
                            }
                    }
                    
                    
                }
                
                
                Spacer()
                
                // MARK: Buttons at the bottom
                HStack {
                    
                    // button to go back in the array or to the last view
                    if sliderArray.count > 1 {
                        Button(action: {
                            goBackward()
                        }, label: {
                            backwardButton
                        })
                        .padding(.trailing, 20)
                    }
                    
                    // button to go forward
                    Button(action: {
                        goForward()
                    }, label: {
                        forwardButton
                    })
                    
                }
                .frame(width: (bounds.size.width - 48))
                .padding(.bottom, 40)
                .offset(y: !notchPhone && onBoardingUsage ? -55 : 0)
            }
            
            .frame(width: bounds.size.width, height: bounds.size.height)
            .background(
                Image("background")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                
            )
            
        }
    }
    
    
    // MARK: -
    var iconCircle: some View {
        
        // MARK: Circle which shows an icon
        ZStack {
            
            // Background Blur
            BlurView(style: .systemUltraThinMaterial)
                .frame(width: 91, height: 91)
                
                // Stroke to get glass effect
                .overlay(
                    Circle()
                        .stroke(
                            LinearGradient(
                                gradient: Gradient(stops: [
                                                    .init(color: Color(#colorLiteral(red: 0.7791666388511658, green: 0.7791666388511658, blue: 0.7791666388511658, alpha: 0.949999988079071)), location: 0),
                                                    .init(color: Color(#colorLiteral(red: 0.7250000238418579, green: 0.7250000238418579, blue: 0.7250000238418579, alpha: 0)), location: 1)]),
                                startPoint: UnitPoint(x: 0.9016393067273221, y: 0.10416647788375455),
                                endPoint: UnitPoint(x: 0.035519096038869824, y: 0.85416653880629)),
                            lineWidth: 0.5
                        )
                )
                .clipShape(Circle())
                .shadow(color: Color.black.opacity(0.2), radius: 5, x: 0, y: 4)
            
            // Actual icon
            Image(systemName: sliderArray[index].sfSymbol)
                .font(.system(size: 30))
                .foregroundColor(.accentColor)
        }
    }
    
    
    // MARK: -
    var forwardButton: some View {
        
        VStack(spacing: 0.0) {
            Text(sliderArray[index].buttonText)
                .font(.system(size: 30))
                .foregroundColor(.accentColor)
            Capsule()
                .gradientForeground(gradient: secondaryGradient)
                .frame(width: 58, height: 6)
            
        }
        .frame(maxWidth: .infinity)
        .frame(height: 53)
        .modifier(offWhiteShadow(cornerRadius: 14))
        
    }
    
    // MARK: -
    var backwardButton: some View {
        Image(systemName: "chevron.backward")
            .font(.system(size: 30))
            .frame(width: 53, height: 53)
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(lineWidth: 3)
            )
            .foregroundColor(.gray)
            .opacity(0.8)
    }
    
    // MARK: - Functions
    
    // MARK: Function to go one step forward in the onboarding process
    func goForward() {
        if index < sliderArray.count - 1 {
            index += 1
        } else {
            goToNextView.toggle()
        }
    }
    
    // MARK: Function to go one step forward in the onboarding process
    func goBackward() {
        if index > 0 {
            index -= 1
        } else {
            goToLastView.toggle()
        }
    }
}
